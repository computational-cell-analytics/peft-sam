import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


def create_barplot(df):
    # Combine 'vit_b_lm' and 'vit_b_em_organelles' into 'Generalist'
    df['model'] = df['model'].replace({'vit_b_lm': r'$\mu$-SAM', 'vit_b_em_organelles': r'$\mu$-SAM'})
    df['model'] = df['model'].replace({'vit_b': 'SAM'})

    custom_order = ['vanilla', 'generalist', 'full_ft', 'AttentionSurgery', 'adaptformer', 'lora', 'fact', 'ssf', 'BiasSurgery', 'LayerNormSurgery', 'freeze_encoder']

    # Convert the column to a categorical type with the custom order
    df = df.sort_values(by='modality', key=lambda x: x.map(lambda val: custom_order.index(val) if val in custom_order else len(custom_order)))

    # Map modality names to more readable ones
    dataset_mapping = {
        "covid_if": "CovidIF",
        "mitolab_glycolytic_muscle": "MitoLab",
        "platy_cilia": "Platynereis",
        "orgasegment": "OrgaSegment",
        "gonuclear": "GoNuclear",
        "hpa": "HPA",
        "livecell": "LIVECell",
    }
    modality_mapping = {
        "vanilla": "Base Model",
        "generalist": "Base Model",
        "full_ft": "Full Finetuning",
        "lora": "LoRA",
    }
    df['modality'] = df['modality'].replace(modality_mapping)
    df['dataset'] = df['dataset'].replace(dataset_mapping)

    # Custom color palette
    custom_palette = {
        "ais": "#FF8B94",        # Blue
        "ip": "#56A4C4",         # Orange
        "ib": "#82D37E",         # Green
        "single box": "#FFE278", # Red
        "single point": "#9C89E2" # Purple
    }
    base_colors = list(custom_palette.values())
    custom_palette = {benchmark: (base_colors[i], mcolors.to_rgba(base_colors[i], alpha=0.5)) for i, benchmark in enumerate(['ais', 'ip', 'ib', 'single box', 'single point'])}

    # Metrics to plot
    metrics = ['ais', 'ip', 'ib', 'single box', 'single point']

    # Melt the data for grouped barplot
    df_melted = df.melt(
        id_vars=["dataset", "modality", "model"],
        value_vars=metrics,
        var_name="benchmark",
        value_name="value"
    )
    df_melted = df_melted.dropna(subset=["value"])

    # Unique datasets and modalities
    datasets = df_melted["dataset"].unique()
    num_datasets = len(datasets)

    # Create subplots for each dataset
    fig, axes = plt.subplots(nrows=(num_datasets + 1) // 2, ncols=2, figsize=(10, num_datasets * 2), constrained_layout=True)
    axes = axes.flatten()

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        dataset_data = df_melted[df_melted["dataset"] == dataset]

        modalities = dataset_data["modality"].unique()
        group_spacing = 1.5  # Increase this value to add more space between groups
        x_positions = [i * group_spacing for i in range(len(modalities))]

        for pos, modality in enumerate(modalities):
            modality_data = dataset_data[dataset_data["modality"] == modality]

            bottom_values = {benchmark: 0 for benchmark in metrics}

            for benchmark in metrics:
                benchmark_data = modality_data[modality_data["benchmark"] == benchmark]

                for j, model in enumerate(["SAM", r"$\mu$-SAM"]):
                    model_data = benchmark_data[benchmark_data["model"] == model]
                    if not model_data.empty:
                        value = model_data["value"].values[0] if len(model_data["value"].values) > 0 else 0

                        # Plot stacked bar
                        ax.bar(
                            x_positions[pos] + metrics.index(benchmark) * 0.2,  # Shift for benchmarks
                            value,
                            width=0.2,
                            bottom=bottom_values[benchmark],
                            color=custom_palette[benchmark][j],
                        )
                        # Update the bottom value for stacking
                        bottom_values[benchmark] += value

        ax.set_title(f"{dataset}")
        ax.set_xticks([p + 0.4 for p in x_positions])
        ax.set_xticklabels(modalities, ha='center')

    benchmark_legend = [Patch(color=custom_palette[benchmark][0], label=f"{benchmark}") for benchmark in metrics]
    model_legend = [
        Patch(color='black', alpha=1.0, label="SAM (Darker Colors)"),
        Patch(color='black', alpha=0.5, label=r"$\mu$-SAM (Lighter Colors)")
    ]
    handles = benchmark_legend + model_legend
    fig.legend(
        handles=handles, loc='lower center', ncol=10, fontsize=10,
    )
    fig.tight_layout(rect=[0.04, 0.05, 1, 0.99])  # Adjust space for the legend
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.text(x=-0.9, y=3.2, s="Mean Segmentation Accuracy", rotation=90, fontweight="bold", fontsize=12)
    plt.savefig("../../results/figures/single_img_training.png")


if __name__ == "__main__":
    df = pd.read_csv("../../results/single_img_training.csv")
    create_barplot(df)
