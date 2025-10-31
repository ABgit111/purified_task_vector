# To change random seed for data loading, modify RANDOM_SEED variable below
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import sys
sys.path.append('/ibex/user/anb/Model_merging/EMR_Merging-main/merge_vit')
import time
import random
from task_vectors import TaskVector, tade_TaskVector
from eval import eval_single_dataset
from args import parse_arguments
from rank_strategy import vit_rank_search
from collections import defaultdict, OrderedDict
import re 
import pickle


def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def apply_vector(vector, pretrained_checkpoint):
    """Apply a task vector to a pretrained model."""
    with torch.no_grad():
        pretrained_model = torch.load(pretrained_checkpoint)
        new_state_dict = {}
        pretrained_state_dict = pretrained_model.state_dict()
        
        # Determine the device from the vector tensors
        vector_device = next(iter(vector.values())).device
        
        for key in pretrained_state_dict:
            if key not in vector:
                print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                new_state_dict[key] = pretrained_state_dict[key]
                continue
            # Move pretrained tensor to the same device as vector tensor
            pretrained_tensor = pretrained_state_dict[key].to(vector_device)
            new_state_dict[key] = pretrained_tensor + vector[key]
    pretrained_model.load_state_dict(new_state_dict, strict=False)
    
    # Move the entire model to the vector device (GPU)
    pretrained_model = pretrained_model.to(vector_device)
    
    return pretrained_model


def emr_merge(task_vectors):
    # Determine device from the first tensor
    device = next(iter(task_vectors[0].vector.values())).device
    
    sum_param = {}
    n2p = []
    for m in range(len(task_vectors)):
        n2p_temp = task_vectors[m].vector
        n2p_temp = {k: v.to(device) for k, v in n2p_temp.items()}
        n2p.append(n2p_temp)
        for n in n2p_temp:
            if n not in sum_param:
                sum_param[n] = []
            sum_param[n].append(n2p_temp[n])
    sum_param = {k: torch.stack(v, 0).mean(0) for k, v in sum_param.items()}
    vector_unified = {}
    scales = torch.zeros(len(task_vectors), device=device)
    masks = {}
    for n in sum_param:
        masks[n] = []
        flag = (sum_param[n]>0) * 2 - 1
        param_max = torch.zeros_like(n2p[0][n])
        for m in range(len(task_vectors)):
            param = task_vectors[m].vector[n]
            mask = (param * flag) > 0
            masks[n].append(mask)
            param_abs = torch.abs(mask*param)
            param_max = torch.where(param_abs>param_max, param_abs, param_max)
            scales[m] += torch.mean(torch.abs(param))
        vector_unified[n] =  param_max * flag
    new_scales = torch.zeros(len(task_vectors), device=device)
    for m in range(len(task_vectors)):
        for n in vector_unified:
            p = vector_unified[n] * masks[n][m]
            new_scales[m] += torch.mean(torch.abs(p))
    rescalers = scales / new_scales

    return vector_unified, masks, rescalers



def create_simple_trainer(model, args, dataset_name, seed=None):
    """Create a simple trainer-like object for the tade_TaskVector."""
    class SimpleTrainer:
        def __init__(self, model, args, dataset_name, seed=None):
            self.model = model
            self.args = args
            self.dataset_name = dataset_name
            self.seed = seed
            
        def get_train_dataloader(self):
            from datasets.registry import get_dataset
            from datasets.common import get_dataloader
            
            dataset = get_dataset(self.dataset_name, self.model.val_preprocess, 
                                location=self.args.data_location, batch_size=self.args.batch_size)
            return get_dataloader(dataset, is_train=True, args=self.args, image_encoder=None)
    
    return SimpleTrainer(model, args, dataset_name, seed)


# Set random seed for reproducibility
def set_random_seed(seed):
    """Set random seed for all random number generators."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set random seed
RANDOM_SEED = 44
set_random_seed(RANDOM_SEED)



exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
model = 'ViT-B-32' # ViT-B-32 | ViT-B-16 | ViT-L-14
args = parse_arguments()
args.home = '/ibex/user/anb/Model_merging/EMR_Merging-main/merge_vit'
args.data_location = args.home + '/data'
args.model = model
args.save = args.home + '/checkpoints/' + model
args.logs_path = args.home + '/logs/' + model
args.batch_size = 16
args.device = 'cuda:0'
args.num_examples = 4096
args.rank_scale = [0.984375, 0.984375, 0.984375, 0.984375, 0.984375, 0.984375, 0.984375, 0.984375]
pretrained_checkpoint = args.home + '/checkpoints/'+model+'/zeroshot.pt'

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_tade_emr_merging.txt'.format(str_time_))

# Load pretrained model
pretrained_model = torch.load(pretrained_checkpoint)

models_for_rank_search = []
for dataset in exam_datasets:
    finetuned_checkpoint = args.home + '/checkpoints/'+model+'/'+dataset+'/finetuned.pt'
    if model == 'ViT-B-16' and dataset == 'Cars':
        models_for_rank_search.append(pickle.load(open(finetuned_checkpoint, 'rb')))
    else:
        models_for_rank_search.append(torch.load(finetuned_checkpoint))
trainers_for_rank_search = [create_simple_trainer(m, args, exam_datasets[i] if i > 0 else exam_datasets[0], seed=RANDOM_SEED) for i, m in enumerate(models_for_rank_search)]

ids = [0,1,2,3,4,5,6,7]

models_for_rank_search_real = [models_for_rank_search[i] for i in ids]
trainers_for_rank_search_real = [trainers_for_rank_search[i] for i in ids]


merge_start = time.perf_counter()
ranks = vit_rank_search(
    models_to_merge=models_for_rank_search_real,
    trainers=trainers_for_rank_search_real,
    exclude_param_names_regex=[],
    rank_ratio=args.rank_scale[0],
    num_ada_examples=[args.num_examples] * 8,
    args=args,
    dataset_name=exam_datasets,
    model_name=model
)


layer_ranks_by_model = defaultdict(dict)
        
unique_layer_names: set[str] = set()
for model_id, layer_name, rank in ranks:
    unique_layer_names.add(layer_name)

id_mapping = {rank_model_id: actual_id for rank_model_id, actual_id in enumerate(ids)}
rank_ratio = [0] * 8  # Initialize list of length 8


for model_id in range(0, 8):
    for layer_name in unique_layer_names:
        if model_id < len(models_for_rank_search):
            model_iter = models_for_rank_search[model_id]
            # Find the layer in the model to get its rank
            for name, module in model_iter.named_modules():
                if name == layer_name and hasattr(module, 'weight'):
                    rank = min(module.weight.shape[0], module.weight.shape[1])
                    layer_ranks_by_model[model_id][layer_name] = rank
                    break


excluded_ranks_sum = 0
for layer_name in unique_layer_names:
    if layer_name in layer_ranks_by_model[1]:
        # Check if this layer should be excluded based on exclude_param_names_regex
        excluded_ranks_sum += layer_ranks_by_model[1][layer_name]


# Update with actual ranks for IDs that are in the ids list
for rank_model_id, layer_name, rank in ranks:
    actual_model_id = id_mapping[rank_model_id]
    layer_ranks_by_model[actual_model_id][layer_name] = rank

# Use the new structure
layer_ranks_by_model = layer_ranks_by_model

##########################################################
# Compute rank ratio for each model individually

# Calculate rank ratio for each model (index 0-7 corresponds to model_id 1-8)
for model_id in range(0, 8):
    model_ranks_sum = 0
    for layer_name in unique_layer_names:
        if layer_name in layer_ranks_by_model[model_id]:
            model_ranks_sum += layer_ranks_by_model[model_id][layer_name]
    
    # rank_ratio[model_id-1] = model_ranks_sum / excluded_ranks_sum
    rank_ratio[model_id] = model_ranks_sum / excluded_ranks_sum

print(f"Rank ratios: {[f'{r:.4f}' for r in rank_ratio]}")
print(f"Excluded ranks sum: {excluded_ranks_sum}")






# Create task vectors using tade_TaskVector
task_vectors = []

iteration = 0
for dataset_name in exam_datasets:
    finetuned_checkpoint = args.home + '/checkpoints/'+model+'/'+dataset_name+'/finetuned.pt'
    
    if model == 'ViT-B-16' and dataset_name == 'Cars':
        finetuned_model = pickle.load(open(finetuned_checkpoint, 'rb'))
    else:
        finetuned_model=torch.load(finetuned_checkpoint)
    
    # Create simple trainers for both models with dataset-specific seeds
    dataset_seed = RANDOM_SEED
    trainer_pretrained = create_simple_trainer(pretrained_model, args, dataset_name, seed=dataset_seed)
    trainer_finetuned = create_simple_trainer(finetuned_model, args, dataset_name, seed=dataset_seed)
    
    # Create tade_TaskVector
    task_vector = tade_TaskVector(
        pretrained_model=pretrained_model,
        finetuned_model=finetuned_model,
        exclude_param_names_regex=[],  # No exclusions for now
        trainer_pretrained=trainer_pretrained,
        trainer_finetuned=trainer_finetuned,
        num_examples=args.num_examples,  # Number of examples to collect
        rank_scale=rank_ratio[iteration],  # Rank scaling factor
        dataset_name=dataset_name,
        model_name=model,
        args=args
    )
    iteration += 1
    # Convert to regular TaskVector format for compatibility with emr_merge
    regular_task_vector = TaskVector(vector=task_vector.get_task_vector())
    task_vectors.append(regular_task_vector)


merge_elapsed_s = time.perf_counter() - merge_start
log.info(f'PAVE time: {merge_elapsed_s:.3f} s')

# Merge models using EMR
vector_unified, masks, rescalers = emr_merge(task_vectors)

# Evaluate on each dataset
accs = []
for i, dataset in enumerate(exam_datasets):
    task_vector_recon = {}
    for n in vector_unified:
        task_vector_recon[n] =  vector_unified[n] * masks[n][i] * rescalers[i]
    image_encoder = apply_vector(task_vector_recon, pretrained_checkpoint)
    metrics = eval_single_dataset(image_encoder, dataset, args)
    log.info(str(dataset) + ':' + str(metrics.get('top1')*100)+'%')
    accs.append(metrics.get('top1')*100)
log.info('Avg ACC:' + str(np.mean(accs)) + '%')
