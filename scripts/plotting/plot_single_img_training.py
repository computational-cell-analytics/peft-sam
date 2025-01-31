from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import os

dataset_mapping = {
    "covid_if": "Covid-IF",
    "orgasegment": "OrgaSegment",
    "gonuclear": "GoNuclear",
    "hpa": "HPA",
    "mitolab_glycolytic_muscle": "MitoLab",
    "platy_cilia": "Platynereis",
}


def get_cellseg1(dataset, model):
    reverse_mapping = {v: k for k, v in dataset_mapping.items()}
    dataset = reverse_mapping[dataset]
    model = model.replace(r"$\mu$SAM", "vit_b_em_organelles") if dataset in ["mitolab_glycolytic_muscle", "platy_cilia"] else model.replace(r"$\mu$SAM", "vit_b_lm")
    model = model.replace("SAM", "vit_b")
    data_path = f"/scratch/usr/nimcarot/sam/experiments/peft/cellseg1/{model}/{dataset}/results/amg_opt.csv"
    amg = 0
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        amg = df['mSA'].iloc[0] if "mSA" in df.columns else None
    return amg


def create_barplot(df):
    # Combine 'vit_b_lm' and 'vit_b_em_organelles' into 'Generalist'
    df['model'] = df['model'].replace({'vit_b_lm': r'$\mu$SAM', 'vit_b_em_organelles': r'$\mu$SAM'})
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
    df['dataset'] = df['dataset'].replace(dataset_mapping)

    df = df[df['dataset'] != 'LIVECell']
    # Custom color palette
    custom_palette = {
        "ais": "#FF8B94",          # Blue
        "ip": "#56A4C4",           # Orange
        "ib": "#82D37E",           # Green
        "single box": "#FFE278",   # Red
        "single point": "#9C89E2"  # Purple
    }
    base_colors = list(custom_palette.values())
    custom_palette = {benchmark: (base_colors[i], mcolors.to_rgba(base_colors[i], alpha=0.5))
                      for i, benchmark in enumerate(['ais', 'ip', 'ib', 'single box', 'single point'])}

    # Metrics to plot
    metrics = ['ais', 'single point', 'ip', 'single box', 'ib']

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

    # Create subplots for each dataset
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), constrained_layout=True)
    axes = axes.flatten()
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        dataset_data = df_melted[df_melted["dataset"] == dataset]

        modalities = list(modality_mapping.values())[1:]
        group_spacing = 2.7  # Increase this value to add more space between groups
        x_positions = [i * group_spacing for i in range(len(modalities))]

        bar_width = 0.35  # Width for each model's bar

        for pos, modality in enumerate(modalities):
            modality_data = dataset_data[dataset_data["modality"] == modality]
            for benchmark_idx, benchmark in enumerate(metrics):
                benchmark_data = modality_data[modality_data["benchmark"] == benchmark]
                SAM_data = benchmark_data[benchmark_data['model'] == 'SAM']
                mu_SAM_data = benchmark_data[benchmark_data['model'] == r'$\mu$SAM']
                SAM_value = SAM_data['value'].values[0] if not SAM_data.empty else 0
                mu_SAM_value = mu_SAM_data['value'].values[0] if not mu_SAM_data.empty else 0
                if SAM_value > mu_SAM_value:
                    models = ['SAM', r'$\mu$SAM']
                else:
                    models = [r'$\mu$SAM', 'SAM']

                for _, model in enumerate(models):
                    cellseg1 = get_cellseg1(dataset, model)
                    model_data = benchmark_data[benchmark_data["model"] == model]
                    if not model_data.empty:
                        value = model_data["value"].values[0] if len(model_data["value"].values) > 0 else 0

                        hatch = "///" if model == r"$\mu$SAM" else None  # Add hatch pattern for $\mu$-SAM
                        linestyle = "--" if model == r"$\mu$SAM" else "-"  # Add linestyle for SAM
                        # Plot non-stacked bar
                        ax.axhline(y=cellseg1, color='black', linestyle=linestyle, linewidth=1)
                        ax.bar(
                            x_positions[pos] + benchmark_idx * bar_width, # + (j - 0.5) * bar_width,
                            value,
                            width=bar_width,
                            facecolor=custom_palette[benchmark][0],
                            hatch=hatch,
                            edgecolor='black',  # Optional: Adds border for better visibility
                        )

        ax.set_title(f"{dataset}", fontsize=15)
        ax.set_xticks([p + 0.7 for p in x_positions])
        ax.tick_params(axis='x', rotation=45)
        ax.set_xticklabels(modalities, ha='center', fontsize=13)

    # Updated legend with hatching and horizontal lines
    benchmark_legend = [Patch(color=custom_palette[benchmark][0], label=f"{benchmark}") for benchmark in metrics]
    model_legend = [
        Patch(facecolor='white', edgecolor='black', hatch=None, label="SAM"),
        Patch(facecolor='white', edgecolor='black', hatch='///', label=r"$\mu$SAM"),
    ]
    line_legend = [
        Line2D([0], [0], color='black', linestyle='-', label="CellSeg1 - SAM"),
        Line2D([0], [0], color='black', linestyle='--', label="CellSeg1 - "+r"$\mu$SAM"),
    ]
    handles = benchmark_legend + model_legend + line_legend
    metric_names = ['AIS', 'Point', r'$I_{\mathbfit{P}}$', 'Box', r'$I_{\mathbfit{B}}$']

    labels = metric_names + ['SAM', r'$\mu$SAM', 'CellSeg1 (SAM)', 'CellSeg1 '+r'($\mu$SAM)']
    fig.legend(
        handles=handles, labels=labels, loc='lower center', ncol=10, fontsize=13,
        bbox_to_anchor=(0.53, 0)
    )
    fig.tight_layout(rect=[0.04, 0.03, 1, 0.98])  # Adjust space for the legend

    plt.text(x=-33, y=0.45, s="Mean Segmentation Accuracy", rotation=90, fontweight="bold", fontsize=18)
    plt.savefig("../../results/figures/single_img_training.png", dpi=300)


if __name__ == "__main__":
    df = pd.read_csv("../../results/single_img_training.csv")
    create_barplot(df)