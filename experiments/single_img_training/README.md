# Single Image Finetuning

Experiments where training is conducted on a single image for microscopy, exploring the impact of extremely limited training data.

## Finetuning Code

``` python
python single_img_finetuning.py --dataset <DATASET_NAME> --model_type <MODEL_TYPE> --save_root <SAVE_PATH> --input_path <DATA_PATH> --learning_rate <LR> --peft_method <PEFT_METHOD>
```

Important Arguments
- `--dataset (-d)`: Specify the dataset to fine-tune on (e.g., orgasegment, covid_if, livecell).
- `--model_type (-m)`: Choose the model type (vit_b, vit_l, vit_h).
- `--save_root (-s)`: Directory to save the checkpoints and logs.
- `--input_path (-i)`: Path to the dataset location.
- `--learning_rate`: Set the learning rate for training.
- `--peft_method`: Specify the PEFT method (e.g., lora, adaptformer).
- `--peft_rank`: Rank value for LoRA.
- `--alpha`: Scaling factor for PEFT methods.
- `--projection_size`: Projection size (for AdaptFormer).
- `--freeze`: Specify which model parts to freeze.
- `--checkpoint_name`: Custom checkpoint name.
- `--export_path (-e)`: Where to save the fine-tuned model for further use.


Training and evaluation jobs can be automatically submitted to recreate the paper's experiments with `python submit_single_img_training.py` and `python submit_single_img_evaluations.py`