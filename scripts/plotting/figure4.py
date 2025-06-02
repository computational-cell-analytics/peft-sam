from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import os
import itertools

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
    data_path = f"/mnt/lustre-grete/usr/u12094/experiments/peft/cellseg1/{model}/{dataset}/results/amg_opt.csv"
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
        "lora_late": "Late LoRA",
        "qlora_late": "Late QLoRA",
        "ClassicalSurgery_late": "Late Ft",
        "full_ft": "Full Ft",
    }
    df['modality'] = df['modality'].replace(modality_mapping)
    dataset_mapping = MEDICO_DATASET_MAPPING if medical else MICROSCOPY_DATASET_MAPPING
    df['dataset'] = df['dataset'].replace(dataset_mapping)

    gen_model = "MedicoSAM" if medical else r"$\mu$SAM"

    df = df[df['dataset'] != 'LIVECell']

    custom_palette = {
        "ais": "#045275",
        "single point": "#7CCBA2",
        "single box": "#90477F",
    }
    base_colors = list(custom_palette.values())
    custom_palette = {benchmark: (base_colors[i], mcolors.to_rgba(base_colors[i], alpha=0.5))
                      for i, benchmark in enumerate(['ais', 'single point', 'single box'])}

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
        'MedicoSAM': '\\\\\\'
    }
    # Create subplots for each dataset
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 15), constrained_layout=True)
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
        ax.tick_params(axis='x', rotation=90)
        ax.set_xticklabels(modalities, ha='center', fontsize=13)

    # Updated legend with hatching and horizontal lines
    benchmark_legend = [Patch(color=custom_palette[benchmark][0], label=f"{benchmark}") for benchmark in metrics]
    model_legend = [
        Patch(facecolor='white', edgecolor='black', hatch=None, label="SAM"),
        Patch(facecolor='white', edgecolor='black', hatch='///', label=r"$\mu$SAM"),
        Patch(facecolor='white', edgecolor='black', hatch='\\\\\\', label="MedicoSAM"),
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
        plt.text(x=-15.5, y=0.74, s="Dice Similarity Coefficient", rotation=90, fontweight="bold", fontsize=18)
    else:
        plt.text(x=-16.5, y=0.55, s="Mean Segmentation Accuracy", rotation=90, fontweight="bold", fontsize=18)

    domain = "medical" if medical else "microscopy"
    plt.savefig(f"../../results/figures/figure4_{domain}.svg", dpi=300)
    plt.savefig(f"../../results/figures/figure4_{domain}.png", dpi=300)

    legend_fig = plt.figure()
    legend_ax = legend_fig.add_axes([0, 0, 1, 1])
    legend_ax.legend(handles, labels, ncol=11, fontsize=13)
    legend_ax.axis('off')
    legend_fig.savefig('../../results/figures/figure4_legend.svg', bbox_inches='tight')


def add_late_data(root, domain):

    results = []
    datasets = MEDICO_DATASET_MAPPING.keys() if domain == "medical" else MICROSCOPY_DATASET_MAPPING.keys()
    modalities = ["lora", "qlora", "ClassicalSurgery"]
    for dataset, modality in itertools.product(datasets, modalities):
        gen_model = "vit_b_medical_imaging" if domain == "medical" else "vit_b_lm"
        gen_model = "vit_b_em_organelles" if dataset in ["mitolab_glycolytic_muscle", "platy_cilia"] else gen_model
        for model in ["vit_b", gen_model]:
            result_dir = os.path.join(root, "late_lora", modality, model, "1_img", "all_matrices", "start_6", dataset, "results")

            instance_segmentation_file = os.path.join(result_dir, "instance_segmentation_with_decoder.csv")
            iterative_start_box_file = os.path.join(result_dir, "iterative_prompting_without_mask", "iterative_prompts_start_box.csv")
            iterative_start_point_file = os.path.join(result_dir, "iterative_prompting_without_mask", "iterative_prompts_start_point.csv")
            # Initialize values to None
            ais, single_box, single_point, ib, ip = None, None, None, None, None

            # Extract 'ais' from instance_segmentation_with_decoder.csv
            if os.path.exists(instance_segmentation_file):
                instance_df = pd.read_csv(instance_segmentation_file)
                ais = instance_df["mSA"].iloc[0] if "mSA" in instance_df.columns else None

            # Extract values for 'iterative_prompts_start_box.csv' (single box and ib)
            if os.path.exists(iterative_start_box_file):
                box_df = pd.read_csv(iterative_start_box_file)
                single_box = box_df["mSA"].iloc[0] if not box_df.empty else None
                ib = box_df["mSA"].iloc[-1] if not box_df.empty else None

            # Extract values for 'iterative_prompts_start_point.csv' (single point and ip)
            if os.path.exists(iterative_start_point_file):
                point_df = pd.read_csv(iterative_start_point_file)
                single_point = point_df["mSA"].iloc[0] if not point_df.empty else None
                ip = point_df["mSA"].iloc[-1] if not point_df.empty else None
            
            # Append the result for the current alpha and rank
            results.append({
                "dataset": dataset,
                "modality": f"{modality}_late",
                "model": model,
                "ais": ais,
                "single box": single_box,
                "single point": single_point,
                "ib": ib,
                "ip": ip
            })
    return pd.DataFrame(results)


if __name__ == "__main__":
    # root = "/scratch/usr/nimcarot/sam/experiments/resource_efficient"
    # df_microscopy = pd.read_csv("../../results/single_img_training_microscopy.csv")
    # df_medical = pd.read_csv("../../results/single_img_training_medical.csv")
    # late_results_medical = add_late_data(root, "medical")
    # late_results_microscopy = add_late_data(root, "microscopy")

    # Combine the late results with the main results
    # df_microscopy = pd.concat([df_microscopy, late_results_microscopy], ignore_index=True)
    # df_medical = pd.concat([df_medical, late_results_medical], ignore_index=True)

    # df_microscopy.to_csv("../../results/single_img_training_microscopy.csv", index=False)
    # df_medical.to_csv("../../results/single_img_training_medical.csv")

    df_microscopy = pd.read_csv("../../results/single_img_training_microscopy.csv")
    df_medical = pd.read_csv("../../results/single_img_training_medical.csv")
    create_barplot(df_microscopy)
    create_barplot(df_medical, medical=True)