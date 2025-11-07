import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_param_names_to_merge, get_modules_to_merge


class TaskVector:
    def __init__(self, pretrained_model: nn.Module = None, finetuned_model: nn.Module = None, exclude_param_names_regex: list = None, task_vector_param_dict: dict = None):
        """
        Task vector. Initialize the task vector from a pretrained model and a finetuned model, or
        directly passing the task_vector_param_dict dictionary.
        :param pretrained_model: nn.Module, pretrained model
        :param finetuned_model: nn.Module, finetuned model
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param task_vector_param_dict: dict, task vector to initialize self.task_vector_param_dict
        """
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
        else:
            self.task_vector_param_dict = {}
            pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
            finetuned_param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(pretrained_param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
            with torch.no_grad():
                for param_name in param_names_to_merge:
                    self.task_vector_param_dict[param_name] = finetuned_param_dict[param_name] - pretrained_param_dict[param_name]

    def __add__(self, other):
        """
        add task vector
        :param other: TaskVector to add, at right side
        :return:
        """
        assert isinstance(other, TaskVector), "addition of TaskVector can only be done with another TaskVector!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert param_name in other.task_vector_param_dict.keys(), f"param_name {param_name} is not contained in both task vectors!"
                new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name] + other.task_vector_param_dict[param_name]
        return TaskVector(task_vector_param_dict=new_task_vector_param_dict)

    def __radd__(self, other):
        """
        other + self = self + other
        :param other: TaskVector to add, at left side
        :return:
        """
        return self.__add__(other)

    def combine_with_pretrained_model(self, pretrained_model: nn.Module, scaling_coefficient: float = 1.0):
        """
        combine the task vector with pretrained model
        :param pretrained_model: nn.Module, pretrained model
        :param scaling_coefficient: float, scaling coefficient to merge the task vector
        :return:
        """
        pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}

        with torch.no_grad():
            merged_params = {}
            for param_name in self.task_vector_param_dict:
                merged_params[param_name] = pretrained_param_dict[param_name] + scaling_coefficient * self.task_vector_param_dict[param_name]

        return merged_params





def reduce_non_diagonal(matrix, reduce_non_diagonal_ratio):
    """
    Reduces the non-diagonal elements of a matrix by a specified ratio.
    
    Args:
        matrix (torch.Tensor): Input matrix to modify
        reduce_non_diagonal_ratio (float): Ratio by which to reduce non-diagonal elements (0 to 1)
        
    Returns:
        torch.Tensor: Matrix with reduced non-diagonal elements
    """
    # Create a copy of the input matrix to avoid modifying the original
    result = matrix.clone()
    
    # Create a diagonal mask (1s on diagonal, 0s elsewhere)
    diagonal_mask = torch.eye(matrix.shape[0], device=matrix.device)
    
    # Create the opposite mask for non-diagonal elements
    non_diagonal_mask = 1 - diagonal_mask
    
    # Multiply non-diagonal elements by (1 - reduce_non_diagonal_ratio)
    result = result * diagonal_mask + result * non_diagonal_mask * (1 - reduce_non_diagonal_ratio)
    
    return result



class Ada_TaskVector_Cholesky:
    def __init__(self, pretrained_model: nn.Module = None, finetuned_model: nn.Module = None, exclude_param_names_regex: list = None, trainer_pretrained = None, trainer_finetuned = None, num_examples: int = None, reduce_non_diagonal_ratio: float = 1, rank_scale: float =1, model_name: str = None, task_vector_param_dict: dict = None, regularization: float = 1e-2):
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
        else:
            self.task_vector_param_dict = {}
            # Ensure both models are on the same device (CPU) for parameter extraction
            pretrained_model.to('cpu')
            finetuned_model.to('cpu')
            pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
            finetuned_param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(pretrained_param_dict.keys()), exclude_param_names_regex=[])
            param_names_to_merge_exclude = get_param_names_to_merge(input_param_names=list(pretrained_param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)

            
            module_types = [nn.Linear]
            ##########################################################
            if model_name == 'gpt2':
                for m in pretrained_model.modules():
                    if isinstance(m, nn.Embedding):       
                        continue
                    if hasattr(m, "weight") and m.weight.dim() == 2:
                        module_types.append(type(m))
                seen, deduped = set(), []
                for t in module_types:
                    if t not in seen:
                        deduped.append(t)
                        seen.add(t)
                module_types = deduped
            ##########################################################
            
            linear_modules_finetuned = get_modules_to_merge(finetuned_model, include_module_types=module_types)

            

            X_finetuned = self.collect_layer_inputs_via_hook(finetuned_model, trainer_finetuned, linear_modules_finetuned, num_examples)



            with torch.no_grad():
                for param_name in param_names_to_merge:
                    if param_name.endswith(".weight") and param_name in param_names_to_merge_exclude:
                        layer_name = param_name[:-len(".weight")]
                        if layer_name in X_finetuned:

                            X_fine = X_finetuned[layer_name]

                            # Move to CPU for covariance computation to save GPU memory
                            X_fine_cpu = X_fine.cpu()
                            C_finetuned = torch.matmul(X_fine_cpu.transpose(0, 1), X_fine_cpu) / num_examples

                            eps = regularization
                            while True:
                                C_finetuned_reduced = C_finetuned + eps * torch.eye(X_fine_cpu.shape[1], device=X_fine_cpu.device)
                                inv_C_finetuned = torch.inverse(C_finetuned_reduced)
                                inv_error = torch.norm(inv_C_finetuned @ C_finetuned_reduced - torch.eye(X_fine_cpu.shape[1], device=X_fine_cpu.device))
                                if inv_error < 1e-2:
                                    break
                                else:
                                    eps = eps * 2

                            if model_name == 'gpt2':
                                W_fine = finetuned_param_dict[param_name]
                                
                                # Move matrices to GPU for SVD computation
                                C_finetuned_gpu = C_finetuned_reduced.to('cuda:0')
                                W_fine_gpu = W_fine.to('cuda:0')
                                
                                # Perform SVD on GPU
                                U_fine, S_fine, Vh_fine = torch.linalg.svd(C_finetuned_gpu @ W_fine_gpu, full_matrices=False)
                                
                                # Move back to CPU for rank computation and truncation
                                
                                r = max(1, round((len(S_fine) * rank_scale)))
                                truncated_SVD_fine_gpu = U_fine[:, :r] @ torch.diag(S_fine[:r]) @ Vh_fine[:r, :]

                                inv_C_finetuned_gpu = inv_C_finetuned.to('cuda:0')
                                
                                W_cr = inv_C_finetuned_gpu @ truncated_SVD_fine_gpu
                                W_task_T = W_cr - pretrained_param_dict[param_name].to('cuda:0')
                                
                                # Move final result to CPU
                                self.task_vector_param_dict[param_name] = W_task_T.cpu()
                                
                                # Clean up GPU memory
                                del C_finetuned_gpu, W_fine_gpu, U_fine, S_fine, Vh_fine, truncated_SVD_fine_gpu, inv_C_finetuned_gpu, W_cr, W_task_T
                                torch.cuda.empty_cache()
                                
                            else:
                                W_fine = finetuned_param_dict[param_name]
                                
                                # Move matrices to GPU for SVD computation
                                C_finetuned_gpu = C_finetuned_reduced.to('cuda:0')
                                W_fine_gpu = W_fine.to('cuda:0')
                                
                                # Perform SVD on GPU
                                U_fine, S_fine, Vh_fine = torch.linalg.svd(C_finetuned_gpu @ W_fine_gpu.transpose(0, 1), full_matrices=False)

                                r = max(1, round((len(S_fine) * rank_scale)))
                                truncated_SVD_fine = U_fine[:, :r] @ torch.diag(S_fine[:r]) @ Vh_fine[:r, :]

                                # Move back to GPU for final computation
                                truncated_SVD_fine_gpu = truncated_SVD_fine.to('cuda:0')
                                inv_C_finetuned_gpu = inv_C_finetuned.to('cuda:0')
                                
                                W_cr = inv_C_finetuned_gpu @ truncated_SVD_fine_gpu
                                W_task_T = W_cr - pretrained_param_dict[param_name].transpose(0, 1).to('cuda:0')
                                
                                # Move final result to CPU
                                self.task_vector_param_dict[param_name] = W_task_T.transpose(0, 1).cpu()
                                
                                # Clean up GPU memory
                                del C_finetuned_gpu, W_fine_gpu, U_fine, S_fine, Vh_fine, truncated_SVD_fine_gpu, inv_C_finetuned_gpu, W_cr, W_task_T
                                torch.cuda.empty_cache()
                        else:
                            # Ensure both tensors are on the same device before subtraction
                            finetuned_tensor = finetuned_param_dict[param_name].to('cpu')
                            pretrained_tensor = pretrained_param_dict[param_name].to('cpu')
                            self.task_vector_param_dict[param_name] = finetuned_tensor - pretrained_tensor
                    else:
                        # Ensure both tensors are on the same device before subtraction
                        finetuned_tensor = finetuned_param_dict[param_name].to('cpu')
                        pretrained_tensor = pretrained_param_dict[param_name].to('cpu')
                        self.task_vector_param_dict[param_name] = finetuned_tensor - pretrained_tensor

    def collect_layer_inputs_via_hook(self, model, trainer, linear_modules, num_examples):
        # Clear GPU memory before moving model to GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        model.to('cuda:0')
        layer_inputs = {name: [] for name in linear_modules.keys()}

        # Ensure model has padding token configured
        if not hasattr(model.config, 'pad_token_id') or model.config.pad_token_id is None:
            raise ValueError("Model must have pad_token_id configured in its config")

        handles = []
        def get_hook(module_name):
            def hook(module, input, output):
                x = input[0].detach().float()
                x = x.reshape(-1, x.shape[-1])
                layer_inputs[module_name].append(x.to('cuda:0'))
            return hook

        for name, module in linear_modules.items():
            handle = module.register_forward_hook(get_hook(name))
            handles.append(handle)

        dataloader = trainer.get_train_dataloader()
        collected_examples = 0
        for inputs in dataloader:
            # Ensure inputs are properly padded when lists are provided; if tensors, assume already padded
            if isinstance(inputs, dict) and 'input_ids' in inputs and isinstance(inputs['input_ids'], list):
                # Add padding token if not present (list-of-ids case)
                if all(isinstance(ids, list) for ids in inputs['input_ids']):
                    max_len = max(len(ids) for ids in inputs['input_ids'])
                    padded_inputs = []
                    for ids in inputs['input_ids']:
                        padding_length = max_len - len(ids)
                        padded_ids = ids + [model.config.pad_token_id] * padding_length
                        padded_inputs.append(padded_ids)
                    inputs['input_ids'] = padded_inputs
                    if 'attention_mask' in inputs and all(isinstance(m, list) for m in inputs['attention_mask']):
                        padded_masks = []
                        for mask in inputs['attention_mask']:
                            padding_length = max_len - len(mask)
                            padded_mask = mask + [0] * padding_length
                            padded_masks.append(padded_mask)
                        inputs['attention_mask'] = padded_masks

            inputs = trainer._prepare_inputs(inputs)
            model(**inputs)
            batch_size = next(iter(layer_inputs.values()))[-1].shape[0]
            collected_examples += batch_size
            if collected_examples >= num_examples:
                break

        for handle in handles:
            handle.remove()

            # Concatenate and trim excess examples
        for name in layer_inputs:
            layer_inputs[name] = torch.cat(layer_inputs[name], dim=0)[:num_examples].to('cuda:0')

        # Move model back to CPU and clear GPU memory
        model.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return layer_inputs
########################################################################################


    def get_task_vector(self):
        return self.task_vector_param_dict
    
    def __add__(self, other):
        """
        add task vector
        :param other: TaskVector to add, at right side
        :return:
        """
        assert isinstance(other, Ada_TaskVector_Cholesky), "addition of TaskVector can only be done with another TaskVector!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert param_name in other.task_vector_param_dict.keys(), f"param_name {param_name} is not contained in both task vectors!"
                new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name] + other.task_vector_param_dict[param_name]
        return Ada_TaskVector_Cholesky(task_vector_param_dict=new_task_vector_param_dict)

    def __radd__(self, other):
        """
        other + self = self + other
        :param other: TaskVector to add, at left side
        :return:
        """
        return self.__add__(other)

    def combine_with_pretrained_model(self, pretrained_model: nn.Module, scaling_coefficient: float = 1.0):
        """
        combine the task vector with pretrained model
        :param pretrained_model: nn.Module, pretrained model
        :param scaling_coefficient: float, scaling coefficient to merge the task vector
        :return:
        """
        pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}

        with torch.no_grad():
            merged_params = {}
            for param_name in self.task_vector_param_dict:
                merged_params[param_name] = pretrained_param_dict[param_name] + scaling_coefficient * self.task_vector_param_dict[param_name]

        return merged_params
    

