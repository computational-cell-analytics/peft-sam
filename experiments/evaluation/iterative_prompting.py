import os

from micro_sam.util import get_sam_model
from micro_sam.evaluation import inference
from micro_sam.evaluation.evaluation import run_evaluation_for_iterative_prompting

from peft_sam.util import get_peft_kwargs

from util import get_paths  # comment this and create a custom function with the same name to run int. seg. on your data
from util import get_default_arguments


def _run_iterative_prompting(dataset_name, exp_folder, predictor, start_with_box_prompt, use_masks):
    prediction_root = os.path.join(
        exp_folder, "start_with_box" if start_with_box_prompt else "start_with_point"
    )
    image_paths, gt_paths = get_paths(dataset_name, split="test")
    inference.run_inference_with_iterative_prompting(
        predictor=predictor,
        image_paths=image_paths,
        gt_paths=gt_paths,
        embedding_dir=None,  # Replace with directory to cache embeddings, eg, 'os.path.join(exp_folder, "embeddings")'
        prediction_dir=prediction_root,
        start_with_box_prompt=start_with_box_prompt,
        use_masks=use_masks
    )
    return prediction_root


def _evaluate_iterative_prompting(dataset_name, prediction_root, start_with_box_prompt, exp_folder):
    _, gt_paths = get_paths(dataset_name, split="test")

    run_evaluation_for_iterative_prompting(
        gt_paths=gt_paths,
        prediction_root=prediction_root,
        experiment_folder=exp_folder,
        start_with_box_prompt=start_with_box_prompt,
    )


def main():
    args = get_default_arguments()

    start_with_box_prompt = args.box  # overwrite to start first iters' prompt with box instead of single point

    peft_kwargs = get_peft_kwargs(
        args.peft_module,
        args.peft_rank,
        alpha=args.alpha,
        dropout=args.dropout,
        projection_size=args.projection_size,
        quantize=args.quantize,
        attention_layers_to_update=args.attention_layers_to_update,
        update_matrices=args.update_matrices,
    )

    # get the predictor to perform inference
    predictor = get_sam_model(model_type=args.model, checkpoint_path=args.checkpoint, peft_kwargs=peft_kwargs)

    prediction_root = _run_iterative_prompting(
        args.dataset, args.experiment_folder, predictor, start_with_box_prompt, args.use_masks
    )
    _evaluate_iterative_prompting(args.dataset, prediction_root, start_with_box_prompt, args.experiment_folder)


if __name__ == "__main__":
    main()
