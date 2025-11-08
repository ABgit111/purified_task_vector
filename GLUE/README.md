# Merging on BLUE benchmarks

## Dependencies and checkpoints 

Please follow [DARE](https://github.com/yule-BUAA/MergeLM) to install the dependencies.

**RoBERTa**: You can download the fine-tuned checkpoints from huggingface [here](https://huggingface.co/vanillaOVO/roberta_base_glue_ckpts/tree/main).

**DeBERTa**: Will be uploaded soon.

```
│cktps/
├──roberta/
│  ├── cola/
│  │  ├── roberta-base_lr1e-05
│  │  │  ├── config.json
│  │  │  ├──......
│  ├── sst2/
│  │  ├── roberta-base_lr1e-05
│  │  │  ├── config.json
│  │  │  ├──......
│  ├── ......
├──deberta/
│  ├── deberta-cola/
│  │  ├── config.json
│  │  ├──......
│  ├── deberta-sst2/
│  │  ├── config.json
│  │  ├──......
│  ├── ......
```

## Merge RoBERTa models

Example of applying PAVE to EMR-Merging:
```{bash}
python pave_merge_roberta_glue.py --merging_method_name pave_emr_merging --num_ada_examples 4096 4096 4096 4096 4096 4096 4096 4096 4096 --language_model_name roberta-base --rank_scale 1.0 0.875 0.875 0.875 0.875 0.875 0.875 0.875 0.875
```

Example of applying PAVE to TIES-Merging:
```{bash}
python pave_methods_glue.py --language_model_name roberta-base --merging_method_name pave_ties_merging --scaling_coefficient 0.9 --num_ada_examples 4096 4096 4096 4096 4096 4096 4096 4096 4096 --rank_scale 1.0 0.984375 0.984375 0.984375 0.984375 0.984375 0.984375 0.984375 0.984375
```

Example of applying PAVE to Task Arithmetic:
```{bash}
python pave_methods_glue.py --language_model_name roberta-base --merging_method_name pave_task_arithmetic --scaling_coefficient 0.3 --num_ada_examples 4096 4096 4096 4096 4096 4096 4096 4096 4096 --rank_scale 1.0 1.0 0.96875 0.96875 0.96875 0.96875 0.96875 0.96875 0.96875
```

## Merge DeBERTa models

Example of applying PAVE to EMR-Merging:
```{bash}
python pave_merge_deberta_glue.py --merging_method_name pave_emr_merging --num_ada_examples 4096 4096 4096 4096 4096 4096 4096 4096 4096 --language_model_name deberta --rank_scale 1.0 0.875 0.875 0.875 0.875 0.875 0.875 0.875 0.875
```

Example of applying PAVE to TIES-Merging:
```{bash}
python pave_methods_glue.py --language_model_name deberta --merging_method_name pave_ties_merging --scaling_coefficient 0.5 --num_ada_examples 4096 4096 4096 4096 4096 4096 4096 4096 4096 --rank_scale 1.0 0.984375 0.984375 0.984375 0.984375 0.984375 0.984375 0.984375 0.984375
```

Example of applying PAVE to Task Arithmetic:
```{bash}
python pave_methods_glue.py --language_model_name deberta --merging_method_name pave_task_arithmetic --scaling_coefficient 0.1 --num_ada_examples 4096 4096 4096 4096 4096 4096 4096 4096 4096 --rank_scale 1.0 0.96875 0.96875 0.96875 0.96875 0.96875 0.96875 0.96875 0.96875
```