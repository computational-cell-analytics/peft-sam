import matplotlib.pyplot as plt
import pandas as pd

import torch

from glob import glob
import os

MEDICO_DATASET_MAPPING = {
    "amd_sd": "AMD-SD",
    "jsrt": "JSRT",
    "mice_tumseg": "Mice TumSeg",
    "papila": "Papila",
    "motum": "MOTUM",
    "psfhs": "PSFHS",
}
MICROSCOPY_DATASET_MAPPING = {
    "livecell": "LIVECell",
    "covid_if": "Covid-IF",
    "orgasegment": "OrgaSegment",
    "gonuclear": "GoNuclear",
    "hpa": "HPA",
    "mitolab_glycolytic_muscle": "MitoLab",
    "platy_cilia": "Platynereis",
}
MODALITY_MAPPING = {
    "freeze_encoder": "Freeze Encoder",
    "LayerNormSurgery": "LN Tune",
    "BiasSurgery": "Bias Tune",
    "ssf": "SSF",
    "fact": "FacT",
    "qlora": "QLoRA",
    "lora": "LoRA",
    "adaptformer": "AdaptFormer",
    "AttentionSurgery": "Attn Tune",
    "full_ft": "Full Ft",
}

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


def prepare_medical_data(df_sam, def_medico):
    # edit the medical imaging training times to the same format as microscopy
    df_sam['model'] = 'SAM'
    def_medico['model'] = 'MedicoSAM'
    df = pd.concat([df_sam, def_medico], axis=0)
    df = df.rename(columns={'best_train_time': 'train_time', 'peft': 'method'})
    # removed faulty qlora inference checkpoints
    df = df[df['dataset'] != 'for_infer']
    df['method'] = df['method'].apply(lambda x: MODALITY_MAPPING[x])
    df['dataset'] = df['dataset'].apply(lambda x: MEDICO_DATASET_MAPPING[x])
    return df


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
        print(path)
        if path.find('cellseg1') != -1 or path.find('for_inference') != -1:
            continue
        model = path.split('/')[-4]
        method = MODALITY_MAPPING[path.split('/')[-3]]
        dataset = MICROSCOPY_DATASET_MAPPING['_'.join(path.split('/')[-2].split('_')[:-1])]
        try:
            # Load checkpoint
            checkpoint = torch.load(path, map_location='cpu')
            training_time = checkpoint.get('train_time', None)

            base_model = "SAM" if model == "vit_b" else r"$\mu$SAM"

            if training_time is not None:
                data.append({'model': base_model, 'method': method, 'dataset': dataset, 'train_time': training_time})
            else:
                print(f"Warning: 'training_time' not found in {path}")
        except Exception as e:
            print(f"Error loading checkpoint {path}: {e}")

    return pd.DataFrame(data)


def barplot(df, is_medical=False):
    if is_medical:
        models = ['SAM', 'MedicoSAM']
    else:
        models = ['SAM', r'$\mu$SAM']
    datasets = df['dataset'].unique()
    modality_order = list(MODALITY_MAPPING.values())
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    handles = []
    labels = []
    # Iterate over models to create a plot for each
    for i, model in enumerate(models):
        ax = axes[i]
        model_data = df[df['model'] == model]
        avg_train_times = model_data.groupby('method')['train_time'].mean().reindex(modality_order)

        for dataset, color in zip(datasets, COLORS):
            dataset_data = model_data[model_data['dataset'] == dataset]
            dataset_data = dataset_data.sort_values(
                by='method', key=lambda x: x.map(lambda val: modality_order.index(val) if val in modality_order
                                                 else len(modality_order))
            )
            ax.plot(
                dataset_data['method'],
                dataset_data['train_time'],
                label=dataset,
                color=color,
                marker='o',
                alpha=0.5
            )
        # Add the average line in dark grey
        ax.plot(
            avg_train_times.index,
            avg_train_times.values,
            color='black',
            marker='o',
            linestyle='--',
            linewidth=2,
            label='Average'
        )
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        # Append to the overall lists (only once)
        if i == 0:
            handles.extend(ax_handles)
            labels.extend(ax_labels)
        # Add title and labels
        ax.set_title(f'Training Times for {model}', fontsize=14)
        if i == 0:
            ax.set_ylabel('Training Time (s)', fontsize=12)  # Only show y-label on the first subplot
        ax.tick_params(axis='x', rotation=45)

    fig.legend(
        handles,
        labels,
        loc='lower center',          # Place legend at the bottom
        ncol=8,                      # 5 columns
        fontsize=10,
        title_fontsize=12
    )
    # Adjust layout
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])  # Adjust space for the legend
    if is_medical:
        plt.savefig('../../results/figures/medical_training_times_v2.pdf', dpi=300)
    else:
        plt.savefig('../../results/figures/training_times_v2.pdf', dpi=300)


def main():
    checkpoint_paths = glob(os.path.join(ROOT, 'checkpoints', '**', 'best.pt'), recursive=True)

    # Extract training times

    if not os.path.exists('../../results/training_times.csv'):
        df = extract_training_times(checkpoint_paths)
        df.to_csv('../../results/training_times.csv')

    df = pd.read_csv('../../results/training_times.csv')
    barplot(df)

    medical_sam_df = pd.read_csv('../../results/medical_imaging_peft_best_times_vit_b.csv')
    medical_medico_sam_df = pd.read_csv('../../results/medical_imaging_peft_best_times_vit_b_medical_imaging.csv')

    medical_df = prepare_medical_data(medical_sam_df, medical_medico_sam_df)
    barplot(medical_df, is_medical=True)


if __name__ == "__main__":
    main()
