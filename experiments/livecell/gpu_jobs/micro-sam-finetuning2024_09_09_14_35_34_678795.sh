#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -p grete:shared
#SBATCH -t 2-00:00:00
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH --constraint=80gb
source activate sam
python ../livecell_finetuning.py -m vit_b -s /scratch/usr/nimcarot/sam/experiments/livecell_peft --peft_method fact --peft_rank 4 