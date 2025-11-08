import copy
import os
import sys
import argparse
from functools import partial
import time
import logging
import json
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from collections import OrderedDict
from utils.glue_data_loader import GLUEDataLoader, glue_data_metrics_map
from utils.metrics import compute_metrics
from utils.customized_trainers import CustomizedTrainer
from utils.utils import set_random_seed
from model_merging_methods.merging_methods import MergingMethod
from inference_plms_glue import dataset_model_learning_rate_mapping_dict, dataset_model_checkpoint_mapping_dict_deberta
from utils.load_config import cache_dir
from model_merging_methods.task_vector import *

parser = argparse.ArgumentParser("Interface for merging roberta models on glue")
parser.add_argument("--language_model_name", type=str, default="roberta-base", help="name of the language model", choices=["roberta-base","deberta"])
parser.add_argument("--merging_method_name", type=str, default="average_merging", help="name of the method to merge models",
                    choices=["average_merging", "task_arithmetic", "fisher_merging", "regmean_merging", "ties_merging", "mask_merging", "ada_ties_merging", "ada_task_arithmetic", "emr_merging"])
parser.add_argument("--scaling_coefficient", type=float, default=1.0, help="scaling coefficient to merge the task vector")
parser.add_argument("--nums_fisher_examples", type=int, nargs="+", help="numbers of examples to compute fisher weights")
parser.add_argument("--fisher_scaling_coefficients", type=float, nargs="+", help="scaling coefficients to merge fisher weights")
parser.add_argument("--normalize_fisher_weight", action="store_true", default=False, help="whether to normalize fisher weights (L2 norm) or not")
parser.add_argument("--minimal_fisher_weight", type=float, default=1e-6, help="the minimal value in fisher weights, used for tackling the potential numerical issues")
parser.add_argument("--nums_regmean_examples", type=int, nargs="+", help="numbers of examples to compute regmean weights")
parser.add_argument("--reduce_non_diagonal_ratio", type=float, default=1.0, help="reduce non-diagonal elements in regmean weights by multiplying this scalar")
parser.add_argument("--param_value_mask_rate", type=float, default=0.8, help="mask rate of the smallest-magnitude parameter values")
parser.add_argument("--weight_format", type=str, help="the format of weights to be masked", default="delta_weight", choices=["finetuned_weight", "delta_weight"])
parser.add_argument("--weight_mask_rate", type=float, default=0.1, help="weight mask rate")
parser.add_argument("--use_weight_rescale", action="store_true", default=False, help="whether to rescale the weight by 1 / (1 - weight_mask_rate)")
parser.add_argument("--mask_strategy", type=str, help="mask strategy", default="random", choices=["random", "magnitude"])
parser.add_argument("--mask_apply_method", type=str, default="average_merging", help="merging method that the mask strategy applies",
                    choices=["average_merging", "task_arithmetic", "fisher_merging", "regmean_merging", "ties_merging", "emr_merging"])
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--gpu", type=int, default=0, help="number of gpu to use")
parser.add_argument("--rank_scale", type=float, nargs="+", help="Scales of ranks to be used for the task vector")

try:
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
except:
    parser.print_help()
    sys.exit()


def task_vector_param_dict_to_single_vector(task_vector):
    task_vector_param_dict = copy.deepcopy(task_vector)
    sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))
    return torch.nn.utils.parameters_to_vector([param.flatten() for param in sorted_task_vector_param_dict.values()])


def get_emr_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer):
    logger.info(f"configuration is {args}")

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to(args.device)
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to(args.device)
    pretrained_model = copy.deepcopy(merged_model)
    pretrained_model.to('cpu')
    pretrained_param_dict = {param_name: param_value for param_name, param_value in
                             pretrained_model.named_parameters()}

    set_random_seed(seed=0)

    Vector_unified, masks, rescales = merging_method.get_merged_model(merged_model=merged_model,
                                                        models_to_merge=models_to_merge,
                                                        exclude_param_names_regex=[".*classifier.*"],
                                                        trainers=trainers,
                                                        scaling_coefficient=args.scaling_coefficient,
                                                        nums_fisher_examples=args.nums_fisher_examples,
                                                        fisher_scaling_coefficients=args.fisher_scaling_coefficients,
                                                        normalize_fisher_weight=args.normalize_fisher_weight,
                                                        minimal_fisher_weight=args.minimal_fisher_weight,
                                                        nums_regmean_examples=args.nums_regmean_examples,
                                                        reduce_non_diagonal_ratio=args.reduce_non_diagonal_ratio,
                                                        param_value_mask_rate=args.param_value_mask_rate,
                                                        weight_format=args.weight_format,
                                                        weight_mask_rates=[args.weight_mask_rate for _ in range(len(models_to_merge))],
                                                        use_weight_rescale=args.use_weight_rescale,
                                                        mask_strategy=args.mask_strategy,
                                                        mask_apply_method=args.mask_apply_method,
                                                        models_use_deepcopy=True,
                                                        rank_scale=args.rank_scale,
                                                        model_name=args.language_model_name)

    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        if args.language_model_name == "deberta":
            checkpoint_num = dataset_model_checkpoint_mapping_dict_deberta[f"{dataset_name}_{args.language_model_name}"]
            merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/deberta/deberta-{dataset_name}/checkpoint-{checkpoint_num}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
            )
        else:
            learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
            merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
            )
        task_vector_recon = {}
        for n in Vector_unified:
            task_vector_recon[n] = Vector_unified[n] * masks[n][idx] * rescales[idx]
        with torch.no_grad():
            merged_params = {}
            for param_name in task_vector_recon:
                merged_params[param_name] = pretrained_param_dict[param_name] + task_vector_recon[param_name]
        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_params:
                param_value.data.copy_(merged_params[param_name])
        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")
    return test_metrics

if __name__ == "__main__":
    args.dataset_names = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]
    assert all([dataset_name in ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"] for dataset_name in args.dataset_names]), \
        'name in dataset_names must be contained in ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]!'
    load_model_paths = []
    for dataset_name in args.dataset_names:
        if args.language_model_name == "deberta":
            checkpoint_num = dataset_model_checkpoint_mapping_dict_deberta[f"{dataset_name}_{args.language_model_name}"]
            load_model_paths.append(f"./ckpts/deberta/deberta-{dataset_name}/checkpoint-{checkpoint_num}")
        else:
            learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
            load_model_paths.append(f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name))
    except:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir)
    glue_data_loader = GLUEDataLoader(tokenizer=tokenizer)

    # load the checkpoint of each individual model that needs to be merged
    models_to_merge, trainers, = [], []
    for dataset_name, load_model_path in zip(args.dataset_names, load_model_paths):
        train_dataset, val_dataset, test_dataset, num_labels = glue_data_loader.load_dataset(dataset_name=dataset_name,
                                                                                             train_split_ratio_for_val=0.1,
                                                                                             max_seq_length=128)
        training_args = TrainingArguments(
            output_dir=load_model_path,                        # load model directory
            per_device_train_batch_size=args.batch_size,       # batch size per device during training
            per_device_eval_batch_size=args.batch_size,        # batch size for evaluation
        )

        assert os.path.exists(os.path.join(training_args.output_dir, "trainer_state.json")), "cannot find file trainer_state.json!"
        model_to_merge = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=training_args.output_dir,
            num_labels=num_labels).to(args.device)
        trainer = CustomizedTrainer(
            model=model_to_merge,               # model to be merged
            args=training_args,                 # training arguments
            train_dataset=train_dataset,        # training dataset
            eval_dataset=test_dataset,          # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),   # function for computing metrics
            tokenizer=tokenizer                 # tokenizer
        )
        models_to_merge.append(model_to_merge.to('cpu'))
        trainers.append(trainer)

    merging_method = MergingMethod(merging_method_name=args.merging_method_name, model_name=args.language_model_name)

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    save_merge_log_path = f"./save_merge_logs/{args.merging_method_name}/{args.language_model_name}"
    os.makedirs(save_merge_log_path, exist_ok=True)

    fh = logging.FileHandler(f"{save_merge_log_path}/{str(time.time())}.log")
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    run_start_time = time.time()
    logger.info(f"********** Run starts. **********")
    performance = get_emr_merge_performance(args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer)



