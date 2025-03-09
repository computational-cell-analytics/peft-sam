from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import os

MICROSCOPY_DATASET_MAPPING = {
    "covid_if": "Covid-IF",
    "orgasegment": "OrgaSegment",
    "gonuclear": "GoNuclear",
    "hpa": "HPA",
    "mitolab_glycolytic_muscle": "MitoLab",
    "platy_cilia": "Platynereis",
}

MEDICO_DATASET_MAPPING = {
    "amd_sd": "AMD-SD",
    "jsrt": "JSRT",
    "mice_tumseg": "Mice TumSeg",
    "papila": "Papila",
    "motum": "MOTUM",
    "psfhs": "PSFHS",
    # "sega": "SegA",
    # "dsad": "DSAD",
    # "ircadb": "IRCADb"
}


def get_cellseg1(dataset, model):
    reverse_mapping = {v: k for k, v in MICROSCOPY_DATASET_MAPPING.items()}
    dataset = reverse_mapping[dataset]
    model = model.replace(r"$\mu$SAM", "vit_b_em_organelles") if dataset in ["mitolab_glycolytic_muscle", "platy_cilia"] else model.replace(r"$\mu$SAM", "vit_b_lm")
    model = model.replace("SAM", "vit_b")
    data_path = f"/scratch-emmy/usr/nimcarot/peft/cellseg1/{model}/{dataset}/results/amg_opt.csv"
    amg = 0
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        amg = df['mSA'].iloc[0] if "mSA" in df.columns else None
    return amg


def create_barplot(df, medical=False):
    # Combine 'vit_b_lm' and 'vit_b_em_organelles' into 'Generalist'
    df['model'] = df['model'].replace({'vit_b_lm': r'$\mu$SAM', 'vit_b_em_organelles': r'$\mu$SAM'})
    df['model'] = df['model'].replace({'vit_b_medical_imaging': 'MedicoSAM'})
    df['model'] = df['model'].replace({'vit_b': 'SAM'})

    custom_order = ['vanilla', 'generalist', 'full_ft', 'AttentionSurgery', 'adaptformer', 'lora', 'fact', 'ssf',
                    'BiasSurgery', 'LayerNormSurgery', 'freeze_encoder']

    # Convert the column to a categorical type with the custom order
    df = df.sort_values(by='modality',
                        key=lambda x: x.map(lambda val: custom_order.index(val)
                                            if val in custom_order else len(custom_order)))

    # Map modality names to more readable ones
    modality_mapping = {
        "vanilla": "Base Model",
        "generalist": "Base Model",
        "freeze_encoder": "Freeze Encoder",
        "lora": "LoRA",
        "qlora": "QLoRA",
        "full_ft": "Full Ft",
    }
    df['modality'] = df['modality'].replace(modality_mapping)
    dataset_mapping = MEDICO_DATASET_MAPPING if medical else MICROSCOPY_DATASET_MAPPING
    df['dataset'] = df['dataset'].replace(dataset_mapping)

    gen_model = "MedicoSAM" if medical else r"$\mu$SAM"

    df = df[df['dataset'] != 'LIVECell']

    custom_palette = {
        "ais": "#045275",
        "point": "#7CCBA2",
        "box": "#90477F",
        "ip": "#089099",
        "ib": "#F0746E",
    }
    base_colors = list(custom_palette.values())
    custom_palette = {benchmark: (base_colors[i], mcolors.to_rgba(base_colors[i], alpha=0.5))
                      for i, benchmark in enumerate(['ais', 'ip', 'ib', 'single box', 'single point'])}

    # Metrics to plot
    # metrics = ['ais', 'single point', 'ip', 'single box', 'ib']
    metrics = ['ais', 'single point', 'single box',]

    # Melt the data for grouped barplot
    df_melted = df.melt(
        id_vars=["dataset", "modality", "model"],
        value_vars=metrics,
        var_name="benchmark",
        value_name="value"
    )
    df_melted = df_melted.dropna(subset=["value"])

    # Unique datasets and modalities
    # datasets = df_melted["dataset"].unique()
    datasets = dataset_mapping.values()

    # dictionairy for hatches to differentiate between models
    hatches = {
        'SAM': '',
        r'$\mu$SAM': '///',
        'MedicoSAM': '\\\\'
    }
    # Create subplots for each dataset
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15), constrained_layout=True)
    axes = axes.flatten()
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        dataset_data = df_melted[df_melted["dataset"] == dataset]

        modalities = list(modality_mapping.values())[1:]
        group_spacing = 1.5 # Increase this value to add more space between groups
        x_positions = [i * group_spacing for i in range(len(modalities))]

        bar_width = 0.35  # Width for each model's bar

        for pos, modality in enumerate(modalities):
            modality_data = dataset_data[dataset_data["modality"] == modality]
            for benchmark_idx, benchmark in enumerate(metrics):
                benchmark_data = modality_data[modality_data["benchmark"] == benchmark]
                SAM_data = benchmark_data[benchmark_data['model'] == 'SAM']
                mu_SAM_data = benchmark_data[benchmark_data['model'] == gen_model]
                SAM_value = SAM_data['value'].values[0] if not SAM_data.empty else 0
                mu_SAM_value = mu_SAM_data['value'].values[0] if not mu_SAM_data.empty else 0
                if SAM_value > mu_SAM_value:
                    models = ['SAM', gen_model]
                else:
                    models = [gen_model, 'SAM']

                for _, model in enumerate(models):
                    if not medical:
                        cellseg1 = get_cellseg1(dataset, model)
                    model_data = benchmark_data[benchmark_data["model"] == model]
                    if not model_data.empty:
                        value = model_data["value"].values[0] if len(model_data["value"].values) > 0 else 0

                        linestyle = "--" if model == gen_model else "-"  # Add linestyle for SAM
                        # Plot non-stacked bar
                        if not medical:
                            ax.axhline(y=cellseg1, color='black', linestyle=linestyle, linewidth=1)
                        ax.bar(
                            x_positions[pos] + benchmark_idx * bar_width, # + (j - 0.5) * bar_width,
                            value,
                            width=bar_width,
                            facecolor=custom_palette[benchmark],
                            hatch=hatches[model],
                            edgecolor='black',  # Optional: Adds border for better visibility
                        )

        ax.set_title(f"{dataset}", fontsize=15)
        ax.set_xticks([p + 0.35 for p in x_positions])
        ax.tick_params(axis='x', rotation=45)
        ax.set_xticklabels(modalities, ha='center', fontsize=13)

    # Updated legend with hatching and horizontal lines
    benchmark_legend = [Patch(color=custom_palette[benchmark][0], label=f"{benchmark}") for benchmark in metrics]
    model_legend = [
        Patch(facecolor='white', edgecolor='black', hatch=None, label="SAM"),
        Patch(facecolor='white', edgecolor='black', hatch='///', label=r"$\mu$SAM"),
        Patch(facecolor='white', edgecolor='black', hatch='\\\\', label="MedicoSAM"),
    ]
    line_legend = [
        Line2D([0], [0], color='black', linestyle='-', label="CellSeg1 - SAM"),
        Line2D([0], [0], color='black', linestyle='--', label="CellSeg1 - "+r"$\mu$SAM"),
    ]
    handles = benchmark_legend + model_legend + line_legend
    # metric_names = ['AIS', 'Point', r'$I_{\mathbfit{P}}$', 'Box', r'$I_{\mathbfit{B}}$']
    metric_names = ['AIS', 'Point', 'Box']

    labels = metric_names + ['SAM', r'$\mu$SAM', 'MedicoSAM', 'CellSeg1 (SAM)', 'CellSeg1 '+r'($\mu$SAM)']

    # fig.legend(
    #    handles=handles, labels=labels, loc='lower center', ncol=10, fontsize=13,
    #    bbox_to_anchor=(0.53, 0)
    # )
    fig.tight_layout(rect=[0.05, 0.03, 1, 0.98])  # Adjust space for the legend
    if medical:
        plt.text(x=-9.5, y=0.74, s="Dice Similarity Coefficient", rotation=90, fontweight="bold", fontsize=18)
    else:
        plt.text(x=-10.5, y=0.55, s="Mean Segmentation Accuracy", rotation=90, fontweight="bold", fontsize=18)

    domain = "medical" if medical else "microscopy"
    plt.savefig(f"../../results/figures/single_img_training_{domain}.svg", dpi=300)
    plt.savefig(f"../../results/figures/single_img_training_{domain}.png", dpi=300)

    legend_fig = plt.figure()
    legend_ax = legend_fig.add_axes([0, 0, 1, 1])
    legend_ax.legend(handles, labels, ncol=11, fontsize=13)
    legend_ax.axis('off')
    legend_fig.savefig('../../results/figures/single_image_legend.svg', bbox_inches='tight')


if __name__ == "__main__":
    df = pd.read_csv("../../results/single_img_training.csv")
    create_barplot(df)

    #df = pd.read_csv("../../results/single_img_training_medical.csv")
    #create_barplot(df, medical=True)