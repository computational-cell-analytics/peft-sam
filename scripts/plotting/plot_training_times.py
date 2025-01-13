import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import torch

from glob import glob
import os

ROOT = "/scratch/usr/nimcarot/sam/experiments/peft"
COLORS = [
    "#FF8B94",  # Original pink
    "#56A4C4",  # Original blue
    "#82D37E",  # Original green
    "#FFE278",  # Original yellow
    "#9C89E2",  # Original purple
    "#FF5722",  # Bold orange
    "#3F51B5",  # Deep indigo
    "#00BCD4",  # Bright cyan
    "#CDDC39",  # Vibrant lime green
]

DATASET_MAPPING = {
    "covid_if": "CovidIF",
    "mitolab_glycolytic_muscle": "MitoLab",
    "platy_cilia": "Platynereis",
    "orgasegment": "OrgaSegment",
    "gonuclear": "GoNuclear",
    "hpa": "HPA",
    "livecell": "LIVECell",
}
MODALITY_MAPPING = {
    "full_ft": "FullFT",
    "lora": "LoRA",
    "AttentionSurgery": "Attn Tune",
    "BiasSurgery": "Bias Tune",
    "LayerNormSurgery": "LN Tuning",
    "fact": "FacT",
    "adaptformer": "AdaptFormer",
    "freeze_encoder": "Freeze Encoder",
    "ssf": "SSF"
}


def extract_training_times(checkpoint_paths):
    """
    Extracts training times from the state dictionaries of a list of checkpoints.

    Args:
        checkpoint_paths (list of str): List of paths to checkpoint files.

    Returns:
        pd.DataFrame: DataFrame with columns ['Checkpoint', 'TrainingTime'].
    """
    data = []

    for path in checkpoint_paths:
        model = path.split('/')[-4]
        method = MODALITY_MAPPING[path.split('/')[-3]]
        dataset = DATASET_MAPPING['_'.join(path.split('/')[-2].split('_')[:-1])]
        try:
            # Load checkpoint
            checkpoint = torch.load(path)
            training_time = checkpoint.get('train_time', None)

            base_model = "SAM" if model == "vit_b" else "microSAM"

            if training_time is not None:
                data.append({'model': base_model, 'method': method, 'dataset': dataset, 'train_time': training_time})
            else:
                print(f"Warning: 'training_time' not found in {path}")
        except Exception as e:
            print(f"Error loading checkpoint {path}: {e}")

    return pd.DataFrame(data)


def barplot(df):
    # Set up a pivot table for easier plotting
    models = df['model'].unique()

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    handles = []
    labels = []
    # Iterate over models to create a plot for each
    for i, model in enumerate(models):
        ax = axes[i]
        model_data = df[df['model'] == model]

        # Create a grouped barplot
        sns.barplot(
            data=model_data,
            x='dataset',
            y='train_time',
            hue='method',
            ax=ax,
            palette=COLORS,
        )
        # Retrieve handles and labels from the plot
        ax_handles, ax_labels = ax.get_legend_handles_labels()

        # Append to the overall lists (only once)
        if i == 0:
            handles.extend(ax_handles)
            labels.extend(ax_labels)
        # Add title and labels
        ax.set_title(f'Training Times for {model}', fontsize=14)
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Training Time (s)', fontsize=12 if i == 0 else 0)  # Only show y-label on the first subplot
        ax.tick_params(axis='x', rotation=45)
        ax.legend_.remove()

    fig.legend(
        handles,
        labels,
        title="Legend",
        loc='lower center',          # Place legend at the bottom
        ncol=5,                      # 5 columns
        fontsize=10,
        title_fontsize=12
    )
    # Adjust layout
    fig.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust space for the legend
    plt.savefig('results/figures/training_times.png', dpi=300)


def main():
    checkpoint_paths = glob(os.path.join(ROOT, 'checkpoints', '**', 'best.pt'), recursive=True)

    # Extract training times

    if not os.path.exists('results/training_times.csv'):
        df = extract_training_times(checkpoint_paths)
        df.to_csv('results/training_times.csv')

    df = pd.read_csv('results/training_times.csv')
    barplot(df)


if __name__ == "__main__":
    main()
