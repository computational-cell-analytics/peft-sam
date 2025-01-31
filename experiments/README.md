# Parameter Efficient Finetuning Methods of Segment Anything for Microscopy.

## Contents

- `param_search`: This includes systematic hyperparameter tuning for LoRA, AdaptFormer, and FACT to optimize performance.
- `peft_multi_datasets`: The core set of experiments that benchmark all PEFT methods on microscopy and medical datasets.
- `single_img_training`: Experiments where training is conducted on a single image for microscopy, exploring the impact of extremely limited training data.
- `evaluation`: This includes all required scripts to run evaluations with a given checkpoint and dataset

## Finetuning Instructions

To finetune SAM using PEFT methods, run the following command:

```bash
python finetuning.py --dataset <DATASET_NAME> --model_type <MODEL_TYPE> --learning_rate <LEARNING_RATE> --n_objects <N_OBJECTS> --save_root <SAVE_DIRECTORY>
```

### Required Arguments:
- `--dataset`: The dataset to use for training.
- `--model_type`: The model type (e.g., vit_b, vit_h, vit_l).
- `--input_path`: Path to the data directory.

### Optional Arguments:
- `--learning_rate`: The learning rate for fine-tuning (default: 1e-5).
- `--n_objects`: Number of objects per batch (default: 25).
- `--save_root`: Directory to save checkpoints and logs.
- `--export_path`: Path to export the fine-tuned model.
- `--freeze`: Parts of the model to freeze during fine-tuning.
- `--peft_method`: The PEFT method to use (LoRA, AdaptFormer, FACT, etc.).
- `--medical_imaging`: Enable fine-tuning on medical imaging datasets.
- `--quantize`: Whether to quantize the model for QLoRA


For a full list of options, run:

```bash
python finetuning.py --help
```

TLDR: Insights on Memory Engagement:
- `vit_b`
    - freeze_encoder: ~33.89 GB
    - QLoRA: ~48.54 GB
    - LoRA: ~48.62 GB
    - FFT: ~49.56 GB

- `vit_h`
    - freeze_encoder: ~36.05 GB
    - QLoRA: ~ 65.68 GB
    - LoRA: ~ 67.14 GB
    - FFT: ~72.34 GB

