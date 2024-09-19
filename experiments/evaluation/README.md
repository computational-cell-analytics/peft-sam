# Segment Anything Evaluation

Scripts for evaluating Segment Anything models and the finetuned `micro_sam` models.

## Evaluation Code

To run the evaluations on your custom dataset, you need to adapt* the scripts a bit.

- `precompute_embeddings.py`: Script to precompute the image embeddings and store it for following evaluation scripts.
- `evaluate_amg.py`: Script to run Automatic Mask Generation (AMG), the "Segment Anything" feature.
- `evaluate_instance_segmentation`: Script to run Automatic Instance Segmentation (AIS), the new feature in micro-sam with added decoder to perform instance segmentation.
- `iterative_prompting.py`: Script to run iterative prompting** (interactive instance segmentation) with respect to the true labels.

Know more about the scripts above and the expected arguments using `<SCRIPT>.py -h`.

TLDR: The most important arguments to be passed are the hinted below:
```bash
python <SCRIPT>.py -m <MODEL_NAME>  # the segment anything model type
                   -c <CHECKPOINT_PATH>  # path to the model checkpoint (default or finetuned models)
                   -e <EXPERIMENT_FOLDER>  # path to store all the evaluations
                   -d <DATASET_NAME>  # not relevant*
                   # arguments relevant for iterative prompting**
                   --box  # starting with box prompt
                   --use_masks  # use logits masks from previous iterations
```

Change the path to your data directory in `util.py`

```
ROOT = "/scratch/usr/nimcarot/data/"
```