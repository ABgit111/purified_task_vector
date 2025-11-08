import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# use the shared helpers
from utils.utils import get_param_names_to_merge, get_modules_to_merge


def collect_layer_inputs_via_hook(
    model: nn.Module,
    trainer,
    linear_modules: Dict[str, nn.Module],
    maximum_number_of_examples: int,
) -> Dict[str, torch.Tensor]:
    """
    Same hook-based input collector used in Ada_TaskVector_Cholesky.
    """
    model.to("cuda:0")
    layer_inputs = {name: [] for name in linear_modules}

    handles = []

    def build_hook(layer_name: str):
        def hook(module, inputs, _output):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            layer_inputs[layer_name].append(x.cpu())
        return hook

    for name, module in linear_modules.items():
        handles.append(module.register_forward_hook(build_hook(name)))

    dataloader = trainer.get_train_dataloader()
    collected = 0
    for batch in dataloader:
        model(**trainer._prepare_inputs(batch))
        collected += next(iter(layer_inputs.values()))[-1].shape[0]
        if collected >= maximum_number_of_examples:
            break

    for h in handles:
        h.remove()

    # concatenate and trim
    for name in layer_inputs:
        layer_inputs[name] = (
            torch.cat(layer_inputs[name], dim=0)[: maximum_number_of_examples]
            .to("cuda:0")
        )

    return layer_inputs


def rank_search(
    models_to_merge: List[nn.Module],
    trainers: List,
    exclude_param_names_regex: List[str],
    rank_ratio: float = 0.5,
    model_name: str | None = None,
    num_ada_examples: List[int] | None = None,
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

        # decide which module types to treat as “linear”
        module_types: List[type] = [nn.Linear]
        if model_name == "gpt2":
            for m in model.modules():
                if (
                    not isinstance(m, nn.Embedding)
                    and hasattr(m, "weight")
                    and m.weight.dim() == 2
                ):
                    module_types.append(type(m))
            # deduplicate while preserving order
            seen, deduped = set(), []
            for t in module_types:
                if t not in seen:
                    deduped.append(t)
                    seen.add(t)
            module_types = deduped

        linear_modules = get_modules_to_merge(model, module_types)

        # gather inputs
        X_per_layer = collect_layer_inputs_via_hook(
            model, trainer, linear_modules, max_ex
        )

        # compute WC singular values
        with torch.no_grad():
            for pname in param_names:
                if not pname.endswith(".weight"):
                    continue
                layer = pname[: -len(".weight")]
                if layer not in X_per_layer:
                    continue

                X = X_per_layer[layer]  # (N, d_in)
                C = (X.T @ X) / max_ex  # (d_in, d_in)
                W = param_dict[pname]

                eps = 1e-1

                while True:
                    C_regularized = C + eps * torch.eye(X.shape[1], device=X.device)
                    inv_C_regularized = torch.inverse(C_regularized)
                    inv_error = torch.norm(inv_C_regularized @ C_regularized - torch.eye(X.shape[1], device=X.device))
                    if inv_error < 5e-2:
                        break
                    else:
                        eps = eps * 2

                L = torch.linalg.cholesky(C_regularized).T

                row_avg_abs = torch.mean(torch.abs(C_regularized), dim=1)  # Shape: (d,)
                # Construct the diagonal matrix
                S_finetuned = torch.diag(row_avg_abs)  # Shape: (d, d)

                # P = W
                # P = L @ W if model_name == "gpt2" else W @ L
                P = C_regularized @ W if model_name == "gpt2" else W @ C_regularized
                # P = S_finetuned @ W if model_name == "gpt2" else W @ S_finetuned


                svals = torch.linalg.svdvals(P).cpu()
                # Normalize each singular value by the sum of 1st to i-th singular values
                # cumulative_sum = torch.cumsum(svals, dim=0)
                # svals = svals / cumulative_sum
                # svals = svals / torch.norm(svals)
                svals = svals / svals.max()

                # cvals = torch.linalg.svdvals(C).cpu()
                # ratio = torch.sqrt(cvals.max())/cvals.min()
                # svals = torch.log(ratio) * svals

                layer_to_model_to_svals[layer][model_id] = svals
                if layer not in layer_shape:
                    shape = (P.shape[1], P.shape[0]) if model_name == "gpt2" else P.shape
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
