# Parameter Efficient Fine-Tuning of Segment Anything Models

`peft-sam` implements several *PEFT (Parameter Efficient Fine-Tuning)* methods for Segment Anything Model (SAM) in the biomedical imaging domain. 

## Installation

How to install `peft_sam` python library from source?

We recommend to first setup an environment with the necessary requirements:

- `environment.yaml`: to set up an environment on any platform.
- `environment_qlora.yaml`: to set up an environment on any platform, with GPU support only.


To create one of these environments and install `peft_sam` into it follow these steps:

1. Clone the repository: `git clone https://github.com/computational-cell-analytics/peft-sam`
2. Enter it: `cd peft-sam`
3. Create the respective environment: `conda env create -f <ENV_FILE>.yaml`
4. Activate the environment: `conda activate peft-sam`
5. Install `peft_sam`: `pip install -e .`

## Citation

If you are using this repository in your research please cite:
- [our preprint](https://doi.org/10.48550/arXiv.2502.00418).
- the [Segment Anything for Microscopy](https://www.nature.com/articles/s41592-024-02580-4) publication.
- and the original [Segment Anything](https://arxiv.org/abs/2304.02643) publication.
- If you use the medical imaging generalist models, please also cite [MedicoSAM](https://doi.org/10.48550/arXiv.2501.11734) publication.
