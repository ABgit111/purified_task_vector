import argparse
import sys
import os
import shutil
import logging
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from typing import List, Dict, Iterable
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from model_merging_methods.merging_methods import MergingMethod
from utils.utils import set_random_seed, smart_tokenizer_and_embedding_resize
from inference_llms_instruct_math_code import create_llm, test_alpaca_eval, test_gsm8k, test_hendrycks_math, test_human_eval, test_mbpp


task_model_mapping_dict = {
    "instruct": "WizardLM-7B-V1.2",
    "math": "WizardMath-7B-V1.0",
    "code": "Llama-2-7b-evolcodealpaca"
}
finetuned_model_backbone_mapping_dict = {
    "WizardLM-7B-V1.2": "Llama-2-7b-hf",
    "WizardMath-7B-V1.0": "Llama-2-7b-hf",
    "Llama-2-7b-evolcodealpaca": "Llama-2-7b-hf"
}

cache_dir = "/ibex/user/anb/Model_merging/DARE/MergeLM/cache_dir"


def get_merge_performance(args: argparse.Namespace, finetuned_model_names: list, merge_task_names: list, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizers: list):
    """
    get the performance of merging method named merging_method_name
    :param args: ArgumentParser, input argument parser
    :param finetuned_model_names: list, names of finetuned models
    :param merge_task_names: list, names of tasks that need to be merged
    :param models_to_merge: list, individual models that need to be merged
    :param trainers: list, trainers of individual models
    :param logger: Logger, logger
    :param merging_method: MergingMethod, the mering method
    :param tokenizers: list of tokenizers
    :return:
    """
    logger.info(f"configuration is {args}")

    try:
        pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.pretrained_model_name), device_map="cpu")
        pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.pretrained_model_name))
    except:
        pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name, cache_dir=cache_dir, device_map="cpu")
        pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name, cache_dir=cache_dir)

    # set the pad_token of pretrained and finetuned tokenizer
    # note that WizardMath-70B-V1.0 adds two tokens {"<pad>": 32000, "[PAD]": 32001} with (32002, 8192) token embedding size
    # therefore, for WizardMath-70B-V1.0, we add one distinct pad_token "<pad>[PAD]" to reshape the token embedding size to (32001, 8192)
    if "WizardMath-70B-V1.0" in finetuned_model_names:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="<pad>[PAD]"),
            model=pretrained_model,
            tokenizer=pretrained_tokenizer,
        )
        for finetuned_model, finetuned_tokenizer in zip(models_to_merge, tokenizers):
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="<pad>[PAD]"),
                model=finetuned_model,
                tokenizer=finetuned_tokenizer,
            )
    else:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            model=pretrained_model,
            tokenizer=pretrained_tokenizer,
        )
        for finetuned_model, finetuned_tokenizer in zip(models_to_merge, tokenizers):
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                model=finetuned_model,
                tokenizer=finetuned_tokenizer,
            )

    print(f"start merging")
    
    # Add memory management for 3x 7B models
    def clear_gpu_memory():
        """Clear GPU memory and run garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()

    class SimpleTextDataset(Dataset):
        def __init__(self, texts: List[str], tokenizer: AutoTokenizer, max_length: int):
            texts = [t for t in texts if isinstance(t, str) and t.strip()]
            if len(texts) == 0:
                raise ValueError("SimpleTextDataset received 0 usable texts after filtering.")
            self.tokenizer = tokenizer
            self.max_length = max_length
            enc = tokenizer(texts, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]

        def __len__(self):
            return self.input_ids.size(0)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx]
            }

    class SimpleTrainer:
        def __init__(self, model, dataset: Dataset, batch_size: int):
            self.model = model
            self.dataset = dataset
            self._train_batch_size = batch_size

        def get_train_dataloader(self):
            return DataLoader(self.dataset, batch_size=self._train_batch_size, shuffle=True)

        def _prepare_inputs(self, inputs: Dict[str, torch.Tensor]):
            device = next(self.model.parameters()).device
            return {k: v.to(device) for k, v in inputs.items()}

    def read_mixed_texts(path: str, max_items: int = None) -> List[Dict]:
        data: List[Dict] = []
        p = Path(path)
        files: List[Path] = []
        if p.is_dir():
            files = sorted(list(p.rglob("*.jsonl"))) + sorted(list(p.rglob("*.parquet")))
        elif p.is_file():
            files = [p]
        else:
            return data

        for fpath in files:
            # Stop if we already have enough
            if max_items is not None and len(data) >= max_items:
                break
            try:
                if fpath.suffix == ".jsonl":
                    with open(fpath, "r", encoding="utf-8") as f:
                        for line in f:
                            if max_items is not None and len(data) >= max_items:
                                break
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            data.append(obj)
                elif fpath.suffix == ".parquet":
                    df = pd.read_parquet(fpath)
                    for _, row in df.iterrows():
                        if max_items is not None and len(data) >= max_items:
                            break
                        data.append(row.to_dict())
            except Exception:
                continue
        return data

    def extract_text(record: Dict) -> str:
        def _extract_string(value) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                parts = []
                for item in value:
                    s = _extract_string(item)
                    if isinstance(s, str) and s.strip():
                        parts.append(s)
                return "\n".join(parts)
            if isinstance(value, dict):
                # Common nested schemas where content is a list of chunks
                for key in ["text", "content", "value", "message"]:
                    if key in value:
                        s = _extract_string(value[key])
                        if isinstance(s, str) and s.strip():
                            return s
                # Fallback: join any string fields
                str_fields = [str(v) for k, v in value.items() if isinstance(v, str) and v.strip()]
                if str_fields:
                    return "\n".join(str_fields)
                return ""
            return str(value) if value is not None else ""

        # Try common fields across datasets
        fields_priority = [
            ("prompt", "response"),
            ("instruction", "output"),
            ("question", "answer"),
            ("problem", "solution"),
        ]
        for a, b in fields_priority:
            a_text = record.get(a)
            b_text = record.get(b)
            if isinstance(a_text, str) and isinstance(b_text, str):
                return a_text + "\n" + b_text
        # Chat-style common schemas
        for chat_key in ["messages", "conversations"]:
            msgs = record.get(chat_key)
            if isinstance(msgs, list) and msgs:
                parts = []
                for m in msgs:
                    if isinstance(m, dict):
                        c = m.get("content") or m.get("text") or m.get("value")
                    else:
                        c = m
                    s = _extract_string(c)
                    if isinstance(s, str) and s.strip():
                        parts.append(s)
                if parts:
                    return "\n".join(parts)
        # Single-field fallbacks often present in parquet
        single_fields = ["text", "prompt", "question", "problem", "completion", "response", "answer", "solution", "output"]
        for f in single_fields:
            v = record.get(f)
            if isinstance(v, str) and v.strip():
                return v
        # Fallback: pick any string fields and join
        strings = [str(v) for k, v in record.items() if isinstance(v, str)]
        return "\n".join(strings) if strings else ""

    def extract_gsm8k_text(record: Dict) -> str:
        q = record.get("question"); a = record.get("answer")
        if isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip():
            return q + "\n" + a
        return extract_text(record)

    def extract_competition_math_text(record: Dict) -> str:
        p = record.get("problem"); s = record.get("solution")
        if isinstance(p, str) and p.strip() and isinstance(s, str) and s.strip():
            return p + "\n" + s
        return extract_text(record)

    def extract_codefeedback_text(record: Dict) -> str:
        # Helper to normalize possibly nested content fields
        def _extract_string(value) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                parts = []
                for item in value:
                    s = _extract_string(item)
                    if isinstance(s, str) and s.strip():
                        parts.append(s)
                return "\n".join(parts)
            if isinstance(value, dict):
                # Common chunk schemas: {"type":"text","text":"..."} or {"text": "..."} or {"content": "..."}
                if "text" in value:
                    return _extract_string(value.get("text"))
                if "content" in value:
                    return _extract_string(value.get("content"))
                if "value" in value:
                    return _extract_string(value.get("value"))
                # Fallback: join any string fields
                str_fields = [str(v) for k, v in value.items() if isinstance(v, str) and v.strip()]
                if str_fields:
                    return "\n".join(str_fields)
                return ""
            return str(value) if value is not None else ""

        # RLHF-like schemas often have chosen/rejected
        if isinstance(record.get("chosen"), str) and record.get("chosen").strip():
            return record["chosen"]
        if isinstance(record.get("rejected"), str) and record.get("rejected").strip():
            return record["rejected"]
        # Prefer pairs if present
        for a, b in [("prompt","response"),("prompt","completion"),("question","answer"),("input","output")]:
            va = record.get(a); vb = record.get(b)
            if isinstance(va, str) and va.strip() and isinstance(vb, str) and vb.strip():
                return va + "\n" + vb
        # Messages: take exactly one round (first user then next assistant)
        msgs = record.get("messages", None)
        if hasattr(msgs, "tolist"):
            try:
                msgs = msgs.tolist()
            except Exception:
                pass
        if not isinstance(msgs, list) or not msgs:
            msgs = record.get("conversations", None)
            if hasattr(msgs, "tolist"):
                try:
                    msgs = msgs.tolist()
                except Exception:
                    pass
        if isinstance(msgs, list) and msgs:
            user_text = None
            assistant_text = None
            # Find first user and the following assistant
            for idx, m in enumerate(msgs):
                if not isinstance(m, dict):
                    continue
                role = m.get("role") or m.get("from")
                content = m.get("content") or m.get("text") or m.get("value")
                if user_text is None and (role == "user" or role == "human" or role == "prompt" or role is None):
                    user_text = _extract_string(content)
                    # search next assistant
                    for n in msgs[idx+1:]:
                        if not isinstance(n, dict):
                            continue
                        nrole = n.get("role") or n.get("from")
                        ncontent = n.get("content") or n.get("text") or n.get("value")
                        if nrole == "assistant" or nrole == "gpt" or nrole == "bot" or nrole == "completion" or nrole is None:
                            assistant_text = _extract_string(ncontent)
                            break
                    break
            if isinstance(user_text, str) and user_text.strip():
                if isinstance(assistant_text, str) and assistant_text.strip():
                    return user_text.strip() + "\n" + assistant_text.strip()
                return user_text.strip()
        # Fallback to generic chat/text extraction
        txt = extract_text(record)
        if isinstance(txt, str) and txt.strip():
            return txt
        # Code-specific fields
        for f in ["code", "solution", "reference_solution", "target", "final_answer", "response", "completion", "output"]:
            v = record.get(f)
            if isinstance(v, str) and v.strip():
                return v
        return ""

    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)
    merged_model = pretrained_model
    
    # Ensure pad_token_id is set in model config
    if not hasattr(merged_model.config, 'pad_token_id') or merged_model.config.pad_token_id is None:
        merged_model.config.pad_token_id = pretrained_tokenizer.pad_token_id
    for model in models_to_merge:
        if not hasattr(model.config, 'pad_token_id') or model.config.pad_token_id is None:
            model.config.pad_token_id = pretrained_tokenizer.pad_token_id
    
    # Build simple trainers if ada methods are requested
    if args.merging_method_name == "pave_task_arithmetic":
        # Build datasets
        # Pretrained uses concatenation of small subsets from each to shape the hook distribution
        code_objs = read_mixed_texts(args.codefeedback_path, max_items=(args.num_ada_examples[2] if args.num_ada_examples else 512))
        # GSM8K folder may contain jsonl or parquet; include any inside the directory tree
        gsm8k_objs = read_mixed_texts(args.gsm8k_path, max_items=(args.num_ada_examples[1] if args.num_ada_examples else 512))
        # MATH dataset under MATH/data; include jsonl/parquet
        comp_objs = read_mixed_texts(args.competition_math_path, max_items=(args.num_ada_examples[1] if args.num_ada_examples else 512))
        # Use specialized extractor for CodeFeedback
        code_texts = [t for t in (extract_codefeedback_text(o) for o in code_objs) if isinstance(t, str) and t.strip()]
        gsm_texts = [t for t in (extract_gsm8k_text(o) for o in gsm8k_objs) if isinstance(t, str) and t.strip()]
        comp_texts = [t for t in (extract_competition_math_text(o) for o in comp_objs) if isinstance(t, str) and t.strip()]
        math_texts = gsm_texts + comp_texts

        if len(code_texts) == 0:
            # Log sample keys to aid debugging
            sample_keys = []
            for obj in code_objs[:5]:
                if isinstance(obj, dict):
                    sample_keys.append(list(obj.keys()))
            logger.error(f"No texts extracted from CodeFeedback at {args.codefeedback_path}. Sample keys from first files: {sample_keys}")
            raise ValueError(f"No texts extracted from CodeFeedback at {args.codefeedback_path}. Check file format/columns.")
        if len(math_texts) == 0:
            raise ValueError(f"No texts extracted from GSM8K/MATH at {args.gsm8k_path} and {args.competition_math_path}.")

        # tokenizers[0] corresponds to first finetuned model; for pretrained, reuse pretrained_tokenizer
        # Trainer 0: pretrained (use a small union to avoid bias)
        pretrained_texts_for_hooks = code_texts[: (args.num_ada_examples[0] if args.num_ada_examples else 512)] + \
                                     math_texts[: (args.num_ada_examples[0] if args.num_ada_examples else 512)]
        ds_pretrained = SimpleTextDataset(pretrained_texts_for_hooks, pretrained_tokenizer, args.max_length)
        trainer_pretrained = SimpleTrainer(pretrained_model, ds_pretrained, args.batch_size)

        # Trainer 1: math finetuned model (WizardMath)
        ds_math = SimpleTextDataset(math_texts[: (args.num_ada_examples[1] if args.num_ada_examples else 512)], tokenizers[0], args.max_length)
        trainer_math = SimpleTrainer(models_to_merge[0], ds_math, args.batch_size)

        # Trainer 2: code finetuned model (Code Alpaca)
        ds_code = SimpleTextDataset(code_texts[: (args.num_ada_examples[2] if args.num_ada_examples else 512)], tokenizers[-1], args.max_length)
        trainer_code = SimpleTrainer(models_to_merge[-1], ds_code, args.batch_size)

        trainers = [trainer_pretrained, trainer_math, trainer_code]

        logger.info(f"ada data sizes -> pretrained:{len(ds_pretrained)} math:{len(ds_math)} code:{len(ds_code)}")
        
        # Print sample data for debugging
        logger.info(f"Sample from ds_pretrained: {pretrained_texts_for_hooks[0][:200]}...")
        logger.info(f"Sample from ds_math: {math_texts[0][:200]}...")
        logger.info(f"Sample from ds_code: {code_texts[0][:200]}...")

        # Default ranks/examples if not provided
        if not args.rank_ratio:
            rank_scale = [0.875, 0.875]
        if not args.num_ada_examples:
            args.num_ada_examples = [len(ds_pretrained), len(ds_math), len(ds_code)]
        rank_scale = [args.rank_ratio, args.rank_ratio, args.rank_ratio]


    print("length of models to merge: ", len(models_to_merge))
    
    # Clear GPU memory before merging
    clear_gpu_memory()

    merged_model = merging_method.get_merged_model(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=['lm_head.*'],
                                                   trainers=trainers,
                                                   scaling_coefficient=args.scaling_coefficient,
                                                   nums_fisher_examples=None,
                                                   fisher_scaling_coefficients=None,
                                                   normalize_fisher_weight=None,
                                                   minimal_fisher_weight=None,
                                                   nums_regmean_examples=None,
                                                   reduce_non_diagonal_ratio=None,
                                                   rank_scale=rank_scale,
                                                   param_value_mask_rate=None,
                                                   weight_format=args.weight_format,
                                                   weight_mask_rates=args.weight_mask_rates,
                                                   use_weight_rescale=args.use_weight_rescale,
                                                   mask_strategy=args.mask_strategy,
                                                   mask_apply_method=args.mask_apply_method,
                                                   num_ada_examples=args.num_ada_examples,
                                                   model_name="llama",
                                                   regularization = args.regularization,
                                                   models_use_deepcopy=False)
    
    # Clear GPU memory after merging
    clear_gpu_memory()
    
    print(f"start to save models...")

    save_math_model_path = save_code_model_path = None
    if args.merge_math:
        save_math_model_path = f"./save_merge_models/{'_'.join(merge_task_names)}/math/{args.save_model_name}"
    if args.merge_code:
        save_code_model_path = f"./save_merge_models/{'_'.join(merge_task_names)}/code/{args.save_model_name}"

    # since the tokenizers of different tasks are different, we need to save them (together with the model) separately
    save_model_paths = [save_math_model_path, save_code_model_path]
    index = 0
    for save_model_path in save_model_paths:
        if save_model_path is not None:
            logger.info(f"saving models at {save_model_path}...")
            merged_model.save_pretrained(save_directory=save_model_path)
            tokenizers[index].save_pretrained(save_directory=save_model_path)
            index += 1
    logger.info(f"models are saved")
    del merged_model, tokenizers


    if save_math_model_path is not None:
        logger.info(f"evaluating merged model on math task...")
        llm = create_llm(finetuned_model_name=save_math_model_path, pretrained_model_name=args.pretrained_model_name,
                         args=args, logger=logger, tensor_parallel_size=args.tensor_parallel_size,
                         just_inference=True, save_model_path=None)
        if args.test_gsm8k:
            test_data_path = "/ibex/user/anb/Model_merging/DARE/MergeLM/math_code_data/gsm8k_test.jsonl"
            test_gsm8k(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                    start_index=args.start_index, end_index=args.end_index, save_model_path=None)
        if args.test_hendrycks_math:
            test_data_path = "/ibex/user/anb/Model_merging/DARE/MergeLM/math_code_data/MATH_test.jsonl"
            test_hendrycks_math(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                                start_index=args.start_index, end_index=args.end_index, save_model_path=None)

    if save_code_model_path is not None:
        logger.info(f"evaluating merged model on code task...")
        llm = create_llm(finetuned_model_name=save_code_model_path, pretrained_model_name=args.pretrained_model_name,
                         args=args, logger=logger, tensor_parallel_size=args.tensor_parallel_size,
                         just_inference=True, save_model_path=None)
        if args.test_human_eval:
            save_gen_results_folder = f"./save_gen_codes_results/{'_'.join(merge_task_names)}/human_eval/{args.save_model_name}"
            test_human_eval(llm=llm, args=args, logger=logger, start_index=args.start_index, end_index=args.end_index,
                            save_model_path=None, save_gen_results_folder=save_gen_results_folder)
        if args.test_mbpp:
            save_gen_results_folder = f"./save_gen_codes_results/{'_'.join(merge_task_names)}/mbpp/{args.save_model_name}"
            test_data_path = "/ibex/user/anb/Model_merging/DARE/MergeLM/math_code_data/mbpp.test.jsonl"
            test_mbpp(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                    start_index=args.start_index, end_index=args.end_index,
                    save_model_path=None, save_gen_results_folder=save_gen_results_folder)

    for save_model_path in save_model_paths:
        if save_model_path is not None:
            shutil.rmtree(save_model_path, ignore_errors=True)
    logger.info(f"inference of merging method {args.merging_method_name} is completed")


parser = argparse.ArgumentParser("Interface for merging LLMs")
parser.add_argument("--merge_instruct", action="store_true", default=False, help="whether to merge instruct model")
parser.add_argument("--merge_math", action="store_true", default=False, help="whether to merge math model")
parser.add_argument("--merge_code", action="store_true", default=False, help="whether to merge code model")
parser.add_argument("--merging_method_name", type=str, default="average_merging", help="name of the method to merge models",
                    choices=["average_merging", "task_arithmetic", "mask_merging", "pave_task_arithmetic"])
parser.add_argument("--scaling_coefficient", type=float, default=1.0, help="scaling coefficient to merge the task vector")
parser.add_argument("--weight_format", type=str, help="the format of weights to be masked", default="delta_weight", choices=["finetuned_weight", "delta_weight"])
parser.add_argument("--weight_mask_rate", type=float, default=0.1, help="weight mask rate")
parser.add_argument("--use_weight_rescale", action="store_true", default=False, help="whether to rescale the weight by 1 / (1 - weight_mask_rate)")
parser.add_argument("--mask_strategy", type=str, help="mask strategy", default="random", choices=["random", "magnitude"])
parser.add_argument("--mask_apply_method", type=str, default="average_merging", help="merging method that the mask strategy applies",
                    choices=["average_merging", "task_arithmetic"])
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--end_index', type=int, default=sys.maxsize)
parser.add_argument("--tensor_parallel_size", type=int, default=1, help="numbers of gpus to use")
parser.add_argument("--rank_ratio", type=float, help="rank scales per finetuned model for ada methods")
parser.add_argument("--regularization", type=float, help="regularization to the covariance matrix")
parser.add_argument("--num_ada_examples", type=int, nargs="+", help="num examples per model for ada methods (pretrained first)")
parser.add_argument("--codefeedback_path", type=str, default="/ibex/user/anb/Model_merging/DARE/MergeLM/math_code_data/CodeFeedback/data", help="path to CodeFeedback dir or jsonl")
parser.add_argument("--gsm8k_path", type=str, default="/ibex/user/anb/Model_merging/DARE/MergeLM/math_code_data/gsm8k/main", help="path to GSM8K dir or jsonl")
parser.add_argument("--competition_math_path", type=str, default="/ibex/user/anb/Model_merging/DARE/MergeLM/math_code_data/MATH/data", help="path to MATH dir or jsonl")
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--test_gsm8k", action="store_true", default=False, help="test on gsm8k")
parser.add_argument("--test_hendrycks_math", action="store_true", default=False, help="test on hendrycks_math")
parser.add_argument("--test_human_eval", action="store_true", default=False, help="test on human_eval")
parser.add_argument("--test_mbpp", action="store_true", default=False, help="test on mbpp")

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit()


if __name__ == "__main__":
    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    assert sum([args.merge_instruct, args.merge_math, args.merge_code]) >= 2, "should merge two tasks at least!"
    finetuned_model_names = []
    merge_task_names = []
    for merge_flag, task_name in zip([args.merge_instruct, args.merge_math, args.merge_code], ["instruct", "math", "code"]):
        if merge_flag:
            finetuned_model_names.append(task_model_mapping_dict[task_name])
            merge_task_names.append(task_name)

    pretrained_model_names = [finetuned_model_backbone_mapping_dict[finetuned_model_name] for finetuned_model_name in finetuned_model_names]
    assert len(set(pretrained_model_names)) == 1, "the backbone of all the finetuned models should be the same!"
    args.pretrained_model_name = pretrained_model_names[0]
    args.weight_mask_rates = [args.weight_mask_rate for _ in range(len(finetuned_model_names))]

    if args.merging_method_name == "average_merging":
        args.save_model_name = f"{args.merging_method_name}"
    elif args.merging_method_name == "task_arithmetic":
        args.save_model_name = f"{args.merging_method_name}_scaling_coefficient_{args.scaling_coefficient}"
    elif args.merging_method_name == "pave_task_arithmetic":
        args.save_model_name = f"{args.merging_method_name}_scaling_coefficient_{args.scaling_coefficient}_rank_ratio_{args.rank_ratio}"
    else:
        assert args.merging_method_name == "mask_merging"
        if args.mask_apply_method == "average_merging":
            mask_apply_method_name = f"{args.mask_apply_method}"
        else:
            assert args.mask_apply_method == "task_arithmetic"
            mask_apply_method_name = f"{args.mask_apply_method}_scaling_coefficient_{args.scaling_coefficient}"
        weight_mask_rates = [str(weight_mask_rate) for weight_mask_rate in args.weight_mask_rates]
        args.save_model_name = f"{args.merging_method_name}/{mask_apply_method_name}/mask_{'_'.join(weight_mask_rates)}_rescale_{args.use_weight_rescale}"

    save_merge_log_path = f"./save_merge_llm_logs/{'_'.join(merge_task_names)}/{args.save_model_name}"
    os.makedirs(save_merge_log_path, exist_ok=True)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(f"{save_merge_log_path}/{str(time.time())}.log")
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    run_start_time = time.time()
    logger.info(f"********** Run starts. **********")

    models_to_merge = []
    finetuned_tokenizers = []
    merging_method = MergingMethod(merging_method_name=args.merging_method_name, model_name="llama")
    for finetuned_model_name in finetuned_model_names:
        finetuned_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name), device_map="cpu")
        finetuned_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name),)
        models_to_merge.append(finetuned_model)
        finetuned_tokenizers.append(finetuned_tokenizer)

    get_merge_performance(args=args, finetuned_model_names=finetuned_model_names, merge_task_names=merge_task_names, models_to_merge=models_to_merge,
                          trainers=[None for _ in range(len(finetuned_model_names))], logger=logger, merging_method=merging_method, tokenizers=finetuned_tokenizers)

    sys.exit()
