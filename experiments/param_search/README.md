# Hyperparameter-Search for PEFT Methods

## LoRA

- Run LoRA finetuning with learning rate 1e-5 on Orgasegment, where $r \in [1, 2, 4, 8, 16, 32, 64]$ and $\alpha \in [1, 2, 4]$
- Run LoRA finetuning with learning rate 1e-5 on Orgasegment, where $r \in [1, 2, 4, 8, 16, 32, 64]$ and $\alpha \in [1, 2, 4]$
- Run LoRA finetuning on OrgaSegment with fixed learning rate = 1e-5 and rank = 32 for $\alpha \in [0.1, 0.25, 0.5, 0.75]$
- Run LoRA finetuning for $\alpha \in [0.1, 1, 8]$ with fixed learning rate 1e-5 and rank 32 on OrgaSegment, Covid-IF, Mitolab and Platynereis.

## AdaptFormer
Run a grid search for AdaptFormer parameters on OrgaSegment
- $\alpha \in [0.1, 0.25, 0.5, 0.75, 1, '\text{learnable scalar}']$
- projection_size $\in [64, 128, 256]$
- dropout $\in [0.1, 0.25, 0.5, None]$

