import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset

# use the shared helpers
from task_vectors import get_param_names_to_merge, get_modules_to_merge


def collect_layer_inputs_via_hook(
    model: nn.Module,
    trainer,
    linear_modules: Dict[str, nn.Module],
    maximum_number_of_examples: int,
    args,
    dataset_name: str,
    seed: int = None,
) -> Dict[str, torch.Tensor]:
    """
    Hook-based input collector adapted for ViT models.
    """
    model.eval()
    model.to("cuda:0")
    layer_inputs = {name: [] for name in linear_modules}
    
    if len(linear_modules) == 0:
        raise RuntimeError("No linear modules provided for hook registration")

    handles = []
    hooks_triggered = {name: 0 for name in linear_modules.keys()}
    
    def get_hook(module_name):
        def hook(module, input, output):
            hooks_triggered[module_name] += 1
            if len(input) > 0 and input[0] is not None:
                x = input[0].detach()
                # For ViT, we need to handle the input shape properly
                if len(x.shape) == 3:  # (batch_size, seq_len, hidden_dim)
                    # Keep the sequence dimension, flatten later if needed
                    x = x.view(-1, x.shape[-1])  # (batch_size * seq_len, hidden_dim)
                elif len(x.shape) == 4:  # (batch_size, channels, height, width)
                    x = x.flatten(1)  # Flatten spatial dimensions
                layer_inputs[module_name].append(x.cpu())
        return hook

    # Register hooks
    for name, module in linear_modules.items():
        handle = module.register_forward_hook(get_hook(name))
        handles.append(handle)

    # Use the dataset for collecting inputs
    dataset = get_dataset(dataset_name, model.val_preprocess, location=args.data_location, batch_size=args.batch_size)
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None, seed=seed)
    
    collected_examples = 0
    model.eval()  # Ensure model is in eval mode
    
    with torch.no_grad():  # Don't need gradients for data collection
        for batch_idx, inputs in enumerate(dataloader):
            inputs = maybe_dictionarize(inputs)
            x = inputs['images'].to('cuda:0')
            
            # Forward pass to trigger hooks
            _ = model(x)
            
            # Count collected examples
            batch_size = x.shape[0]
            collected_examples += batch_size
            
            if collected_examples >= maximum_number_of_examples:
                break

    for handle in handles:
        handle.remove()

    # Find layers with no data
    empty_layers = [name for name in layer_inputs if not layer_inputs[name]]
    if empty_layers:
        # Remove empty layers from the dictionary
        for layer_name in empty_layers:
            del layer_inputs[layer_name]
    
    # Check if we have any data at all
    if not layer_inputs:
        raise RuntimeError("No data collected for any layers!")

    # Concatenate and trim excess examples (keep on CPU to save GPU memory)
    for name in layer_inputs:
        layer_inputs[name] = torch.cat(layer_inputs[name], dim=0)[:maximum_number_of_examples]

    return layer_inputs


def rank_search(
    models_to_merge: List[nn.Module],
    trainers: List,
    exclude_param_names_regex: List[str],
    rank_ratio: float = 0.5,
    model_name: str | None = None,
    num_ada_examples: List[int] | None = None,
    args=None,
    dataset_name: str = None,
) -> List[Tuple[str, str, int]]:
    """
    Determine a rank for every (model, layer) pair by discarding the smallest
    regularised singular values until the total remaining-rank ratio drops
    below `rank_ratio`.
    """
    # store singular values per layer per model
    layer_to_model_to_svals: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    layer_shape: Dict[str, Tuple[int, int]] = {}

    for model_id, (model, trainer, max_ex) in enumerate(
        zip(models_to_merge, trainers, num_ada_examples)
    ):
        model.to("cuda:0")
        param_dict = {n: p for n, p in model.named_parameters()}
        param_names = get_param_names_to_merge(
            list(param_dict.keys()), exclude_param_names_regex
        )

        # For ViT models, we focus on linear layers in the transformer blocks
        module_types: List[type] = [nn.Linear]

        linear_modules = get_modules_to_merge(model, module_types)

        # gather inputs - adapted for ViT
        X_per_layer = collect_layer_inputs_via_hook(
            model, trainer, linear_modules, max_ex, args, dataset_name[model_id], seed=getattr(trainer, 'seed', None)
        )

        # compute WC singular values
        with torch.no_grad():
            for pname in param_names:
                if not pname.endswith(".weight"):
                    continue
                layer = pname[: -len(".weight")]
                if layer not in X_per_layer:
                    continue

                X = X_per_layer[layer]  # (N, d_in) on CPU
                C_cpu = (X.T @ X) / max_ex  # (d_in, d_in) on CPU
                # Free memory of inputs after covariance is computed
                del X
                try:
                    del X_per_layer[layer]
                except Exception:
                    pass
                W = param_dict[pname]

                eps = 1e-1

                while True:
                    C_regularized = C_cpu.to('cuda:0') + eps * torch.eye(C_cpu.shape[0], device='cuda:0')
                    inv_C_regularized = torch.inverse(C_regularized)
                    inv_error = torch.norm(inv_C_regularized @ C_regularized - torch.eye(C_cpu.shape[0], device='cuda:0'))
                    if inv_error < 5e-2:
                        break
                    else:
                        eps = eps * 2

                # For ViT, use the same approach as in tade_TaskVector
                # Check matrix dimensions and handle accordingly
                if W.shape[1] == C_regularized.shape[0]:
                    # Use SVD on the covariance matrix weighted by the weights
                    P = (W.to('cuda:0')) @ C_regularized
                else:
                    # If dimensions don't match, use the regularized covariance directly
                    P = C_regularized @ (W.to('cuda:0'))

                svals = torch.linalg.svdvals(P).cpu()
                # Normalize singular values
                svals = svals / svals.max()

                layer_to_model_to_svals[layer][model_id] = svals
                if layer not in layer_shape:
                    shape = (P.shape[1], P.shape[0])
                    layer_shape[layer] = shape

    # greedy rank trimming
    results: List[Tuple[str, str, int]] = []
    for layer, model2s in layer_to_model_to_svals.items():
        dout, din = layer_shape[layer]
        full_rank = min(dout, din)
        
        
        stopping_rank = full_rank - round((full_rank - round(full_rank * rank_ratio))*3/2)

        # stopping_rank = 0 

        rem = {m: full_rank for m in model2s}
        ptr = {m: full_rank - 1 for m in model2s}

        total_full = full_rank * len(model2s)
        total_rem = total_full

        while total_rem / total_full >= rank_ratio and any(p >= 0 for p in ptr.values()):
            smallest, chosen = float("inf"), None
            for m, s in model2s.items():
                # Only consider models that haven't reached the stopping rank
                if ptr[m] >= 0 and rem[m] > stopping_rank and s[ptr[m]] < smallest:
                    smallest, chosen = s[ptr[m]], m
            
            # If no model can be trimmed (all have reached stopping rank), break
            if chosen is None:
                break
                
            rem[chosen] -= 1
            ptr[chosen] -= 1
            total_rem -= 1

        for m, r in rem.items():
            results.append((m, layer, r))

    return results


def vit_rank_search(
    models_to_merge: List[nn.Module],
    trainers: List,
    exclude_param_names_regex: List[str],
    rank_ratio: float = 0.5,
    num_ada_examples: List[int] = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
    args=None,
    model_name: str = "vit",
    dataset_name: List[str] = ["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"],  # Default dataset for rank search
) -> List[Tuple[str, str, int]]:
    """
    ViT-specific wrapper for rank_search that handles ViT model specifics.
    """
    return rank_search(
        models_to_merge=models_to_merge,
        trainers=trainers,
        exclude_param_names_regex=exclude_param_names_regex,
        rank_ratio=rank_ratio,
        model_name=model_name,  # Set model_name to vit
        num_ada_examples=num_ada_examples,
        args=args,
        dataset_name=dataset_name,
    )
