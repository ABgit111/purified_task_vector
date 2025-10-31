# ViTs

## Dependencies

Please follow [task_vectors](https://github.com/mlfoundations/task_vectors) to install the dependencies.

## Checkpoints and datasets

You can download the fine-tuned checkpoints from the Google Drive [task_vectors_checkpoints](https://drive.google.com/drive/folders/1u_Tva6x0p6oxu5Eo0ZZsf-520Cc_3MKw)

Please follow [EMR-Merging](https://github.com/harveyhuang18/EMR_Merging/tree/main/merge_vit) to download the datasets and checkpoints.


## Eval

Run EMR-Merging
> python main_emr_merging.py

Run EMR-Merging with PAVE
> python 

The recommended setting is num_examples = 4096. If an out-of-memory issue occurs, you may reduce it to num_examples = 1024 for ViT-14-L.





