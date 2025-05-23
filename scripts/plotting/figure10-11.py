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
    "ClassicalSurgery": "Late Ft",
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


def add_late_data(root, domain):
    data = []
    dataset_mapping = MEDICO_DATASET_MAPPING if domain == "medical" else MICROSCOPY_DATASET_MAPPING
    methods = ["ClassicalSurgery"]
    for dataset in dataset_mapping.keys():
        gen_model = "vit_b_medical_imaging" if domain == "medical" else "vit_b_lm"
        gen_model = "vit_b_em_organelles" if dataset in ["mitolab_glycolytic_muscle", "platy_cilia"] else gen_model
        for model in ["vit_b", gen_model]:
            for method in methods:
                path = os.path.join(root, "checkpoints", model, "late_lora", method, "all_matrices", "start_6", f"{dataset}_sam", "best.pt")

            try:
                # Load checkpoint
                checkpoint = torch.load(path, weights_only=False)
                training_time = checkpoint.get('train_time', None)
                if domain == "microscopy":
                    base_model = "SAM" if model == "vit_b" else r"$\mu$SAM"
                else:
                    base_model = "MedicoSAM" if model == "vit_b_medical_imaging" else "SAM"

                if training_time is not None:
                    data.append({'model': base_model, 'method': MODALITY_MAPPING[method], 'dataset': dataset_mapping[dataset], 'train_time': training_time})
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
        plt.savefig('../../results/figures/figure11.pdf', dpi=300)
    else:
        plt.savefig('../../results/figures/figure10.pdf', dpi=300)



def main():

    # late_data = add_late_data(ROOT, "microscopy")
    # training_times_microscopy = pd.read_csv('../../results/training_times_microscopy.csv')

    # Combine the late data with the main training times
    # training_times_microscopy = pd.concat([training_times_microscopy, late_data], axis=0, ignore_index=True)
    # Save the combined dataframe
    # training_times_microscopy.to_csv('../../results/training_times_microscopy.csv', index=False)

    # Load the main results
    training_times_microscopy = pd.read_csv('../../results/training_times_microscopy.csv')
    barplot(training_times_microscopy)

    # medical_sam_df = pd.read_csv('../../results_depricated/medical_imaging_peft_best_times_vit_b.csv')
    # medical_medico_sam_df = pd.read_csv('../../results_depricated/medical_imaging_peft_best_times_vit_b_medical_imaging.csv')

    # medical_df = prepare_medical_data(medical_sam_df, medical_medico_sam_df)
    # late_data = add_late_data(ROOT, "medical")
    # Combine the late data with the main training times
    # medical_df = pd.concat([medical_df, late_data], axis=0, ignore_index=True)

    # Save the combined dataframe
    # medical_df.to_csv('../../results/training_times_medical.csv', index=False)

    medical_df = pd.read_csv('../../results/training_times_medical.csv')
    barplot(medical_df, is_medical=True)


if __name__ == "__main__":
    main()
