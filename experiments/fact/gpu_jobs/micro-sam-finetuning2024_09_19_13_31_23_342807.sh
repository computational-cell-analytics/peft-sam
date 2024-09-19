#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -p grete:shared
#SBATCH -t 2-00:00:00
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH --constraint=80gb
source activate sam
python ../finetuning.py -m vit_b -d livecell -s /scratch/usr/nimcarot/sam/experiments/fact_dropout --peft_rank 4 --peft_method fact 