# Merging on NLG tasks

## Dependencies and checkpoints

### Dependencies

Please follow [DARE](https://github.com/yule-BUAA/MergeLM) to install the dependencies.

### Checkpoints

Base model:[Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
Math reasoning: [WizardMath-7B-V1.0](https://huggingface.co/WizardLMTeam/WizardMath-7B-V1.0)
Code generating: [Llama-2-7b-evolcodealpaca](https://huggingface.co/RedHatAI/Llama-2-7b-evolcodealpaca)



## Scripts for Merging Models

* Example of merging with Task Arithmetic:
```{bash}
python merge_pave_math_code.py \
    --merge_math \
    --merge_code \
    --merging_method_name pave_task_arithmetic \
    --scaling_coefficient 1.0 \
    --tensor_parallel_size 1 \
    --num_ada_examples 512 512 512 \
    --rank_ratio 0.99829102 \
    --test_gsm8k \
    --test_hendrycks_math
    --regularization 0.01
```

* Example of merging with :
```{bash}
python merge_pave_math_code.py \
    --merge_math \
    --merge_code \
    --merging_method_name ada_task_arithmetic \
    --scaling_coefficient 0.5 \
    --tensor_parallel_size 1 \
    --num_ada_examples 512 512 512 \
    --rank_ratio 0.9987793 \
    --test_gsm8k \
    --test_hendrycks_math
    --regularization 0.5
```

## Scripts for evaluation

Please follow [DARE](https://github.com/yule-BUAA/MergeLM) for evaluation on GSM8K, MATH, Human Eval and MBPP.
