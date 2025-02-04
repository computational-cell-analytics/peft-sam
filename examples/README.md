# Example Scripts for `Parameter Efficient Fine-Tuning of Segment Anything Models`

## Finetuning Scripts

Here are the scripts for finetuning Segment Anything Model, using the finetuning scheme supported by `micro-sam`, for several PEFT methods:
- `finetune_sam_using_peft.py` - Finetune Segment Anything Model on OrgaSegment, with annotations for organoid segmentation in brightfield microscopy images, using several PEFT methods.

> NOTE 1: See `python finetune_sam_using_peft.py` for details on all supported PEFT methods.

> NOTE 2: For using the domain-specific generalist models, please refer to their respective repositories: i.e. [`micro-sam`](https://github.com/computational-cell-analytics/micro-sam) and [`medico-sam`](https://github.com/computational-cell-analytics/medico-sam).

## Inference Scripts

> TODO: CLI support coming soon! if you would like to test it meanwhile, please see the [experiments](../experiments/evaluation) folder for details.
