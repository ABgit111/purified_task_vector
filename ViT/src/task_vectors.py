import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
import pickle


def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    """
    get the names of parameters that need to be merged
    :param input_param_names: list, names of input parameters
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge


def get_modules_to_merge(model: nn.Module, include_module_types: list):
    """
    get the model modules that need to be merged, whose type is in include_module_types
    :param model: nn.Module, input model
    :param include_module_types: list, module types that want to include
    :return:
    """
    modules_to_merge = {}
    for module_name, module in model.named_modules():
        is_valid_type = not include_module_types or any([isinstance(module, include_module_type) for include_module_type in include_module_types])
        if is_valid_type:
            modules_to_merge[module_name] = module
    return modules_to_merge


class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None, dataset_name=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                print('TaskVector:' + finetuned_checkpoint)
                pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                # pretrained_state_dict = pickle.load(open(pretrained_checkpoint, 'rb')).state_dict()
                if dataset_name == 'Cars':
                    finetuned_state_dict = pickle.load(open(finetuned_checkpoint, 'rb')).state_dict()
                else:
                    finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                # finetuned_state_dict = pickle.load(open(finetuned_checkpoint, 'rb')).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def weightmerging(self, taskvectors, coefficients):
        with torch.no_grad():
            new_vector = {}
            for key in taskvectors[0].vector:
                new_vector[key] = sum(coefficients[k] * taskvectors[k][key] for k in range(len(taskvectors)))
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model


class tade_TaskVector:
    def __init__(self, pretrained_model: nn.Module = None, finetuned_model: nn.Module = None, exclude_param_names_regex: list = None, trainer_pretrained = None, trainer_finetuned = None, num_examples: int = None, reduce_non_diagonal_ratio: float = 1, rank_scale: float =1, model_name: str = None, task_vector_param_dict: dict = None, args = None, dataset_name: str = None):
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
        else:
            self.task_vector_param_dict = {}
            pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
            finetuned_param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(pretrained_param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)

            # For ViT models, we focus on linear layers in the transformer blocks
            module_types = [nn.Linear]
            
            linear_modules_pretrained = get_modules_to_merge(pretrained_model, include_module_types=module_types)
            linear_modules_finetuned = get_modules_to_merge(finetuned_model, include_module_types=module_types)

            # Collect layer inputs for both models
            X_pretrained = self.collect_layer_inputs_via_hook(pretrained_model, trainer_pretrained, linear_modules_pretrained, num_examples, args, dataset_name, seed=getattr(trainer_pretrained, 'seed', None))
            X_finetuned = self.collect_layer_inputs_via_hook(finetuned_model, trainer_finetuned, linear_modules_finetuned, num_examples, args, dataset_name, seed=getattr(trainer_finetuned, 'seed', None))

            with torch.no_grad():
                for param_name in param_names_to_merge:
                    if param_name.endswith(".weight"):
                        layer_name = param_name[:-len(".weight")]
                        if layer_name in X_finetuned:
                            X_pre = X_pretrained[layer_name]
                            X_fine = X_finetuned[layer_name]

                            C_pretrained = torch.matmul(X_pre.transpose(0, 1), X_pre)/num_examples
                            C_finetuned = torch.matmul(X_fine.transpose(0, 1), X_fine)/num_examples

                            eps = 1e-1

                            while True:
                                C_pretrained_reduced = C_pretrained + eps * torch.eye(X_pre.shape[1], device=X_pre.device)
                                C_finetuned_reduced  = C_finetuned  + eps * torch.eye(X_fine.shape[1], device=X_fine.device)
                                inv_C_finetuned = torch.inverse(C_finetuned_reduced)
                                inv_C_pretrained = torch.inverse(C_pretrained_reduced)
                                inv_error = torch.norm(inv_C_finetuned @ C_finetuned_reduced - torch.eye(X_fine.shape[1], device=X_fine.device)) + torch.norm(inv_C_pretrained @ C_pretrained_reduced - torch.eye(X_pre.shape[1], device=X_pre.device))
                                if inv_error < 5e-2:
                                    break
                                else:
                                    eps = eps * 2

                            # For ViT, we use the same approach as the non-GPT2 case
                            W_fine = finetuned_param_dict[param_name]

                            # Check matrix dimensions and handle accordingly
                            if W_fine.shape[1] == C_finetuned_reduced.shape[0]:
                                # Use SVD on the covariance matrix weighted by the fine-tuned weights
                                U_fine, S_fine, Vh_fine = torch.linalg.svd(W_fine @ C_finetuned_reduced, full_matrices=False)
                            else:
                                # If dimensions don't match, use SVD on the weight matrix directly
                                U_fine, S_fine, Vh_fine = torch.linalg.svd(C_finetuned_reduced @ W_fine, full_matrices=False)

                            r = max(1, round((len(S_fine) * rank_scale)))
                            truncated_SVD_fine = U_fine[:, :r] @ torch.diag(S_fine[:r]) @ Vh_fine[:r, :]

                            if W_fine.shape[1] == C_finetuned_reduced.shape[0]:
                                W_cr = truncated_SVD_fine @ inv_C_finetuned
                            else:
                                W_cr = inv_C_finetuned @ truncated_SVD_fine
                            
                            W_task_T = W_cr - pretrained_param_dict[param_name]

                            self.task_vector_param_dict[param_name] = W_task_T
                        else:
                            # For layers without collected inputs, use simple difference
                            self.task_vector_param_dict[param_name] = finetuned_param_dict[param_name] - pretrained_param_dict[param_name]
                    else:
                        # For non-weight parameters, use simple difference
                        self.task_vector_param_dict[param_name] = finetuned_param_dict[param_name] - pretrained_param_dict[param_name]

    def collect_layer_inputs_via_hook(self, model, trainer, linear_modules, num_examples, args, dataset_name, seed=None):
        model.eval()  # Set model to eval mode
        model.to('cuda:0')
        layer_inputs = {name: [] for name in linear_modules.keys()}
        
        # print(f"Collecting inputs for {len(linear_modules)} linear modules:")
        # for name in linear_modules.keys():
        #     print(f"  - {name}")
            
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
                    # print(f"Hook triggered for {module_name}: input shape {x.shape}")
            return hook

        # Register hooks
        for name, module in linear_modules.items():
            handle = module.register_forward_hook(get_hook(name))
            handles.append(handle)
            # print(f"Registered hook for {name}")

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
                
                # print(f"Collected {collected_examples}/{num_examples} examples")
                
                if collected_examples >= num_examples:
                    break

        for handle in handles:
            handle.remove()

            
        # Find layers with no data
        empty_layers = [name for name in layer_inputs if not layer_inputs[name]]
        if empty_layers:
            for layer_name in empty_layers:
                del layer_inputs[layer_name]
        
        # Check if we have any data at all
        if not layer_inputs:
            raise RuntimeError("No data collected for any layers!")

        # Concatenate and trim excess examples (keep on CPU to save GPU memory)
        for name in layer_inputs:
            layer_inputs[name] = torch.cat(layer_inputs[name], dim=0)[:num_examples].to('cuda:0')

        return layer_inputs

    def get_task_vector(self):
        return self.task_vector_param_dict
    
    def __add__(self, other):
        """
        add task vector
        :param other: TaskVector to add, at right side
        :return:
        """
        assert isinstance(other, tade_TaskVector), "addition of TaskVector can only be done with another TaskVector!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert param_name in other.task_vector_param_dict.keys(), f"param_name {param_name} is not contained in both task vectors!"
                new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name] + other.task_vector_param_dict[param_name]
        return tade_TaskVector(task_vector_param_dict=new_task_vector_param_dict)

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