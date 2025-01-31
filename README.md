# Parameter Efficient Finetuning of Segment Anything Models

PEFT-SAM is a research project exploring **PEFT (Parameter Efficient Fine-Tuning)** methods for **Segment Anything** (SAM) in the **biomedical imaging** domain. 

## Installation

To install and run `peft_sam`, follow the steps below.

### Step 1: Set Up Environment

We recommend creating a new virtual environment to avoid conflicts with existing packages. To do this, you can use `conda` or `pip` to create the environment.

1. **Create a virtual environment:**
```bash
conda env create -f environment.yaml
```
2. Activate the environment
```bash
conda activate peft-sam
```
3. Install the package from source
```bash
git clone https://github.com/yourusername/peft-sam.git
cd peft-sam
pip install -e .
```
