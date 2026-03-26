import tools.hardware_test as ht
from tools.configuration_generator import config
import sys
import argparse
import importlib
import inspect
from tools.config_validator import validate_config

ACTIONS = {
    "train": ("train.train", "run"),
    "eval": ("validation.validation", "run"),
    "validation": ("validation.validation", "run"),
    "test": ("test.test", "run"),
}

def _resolve_action_func(action: str):
    module_name, func_name = ACTIONS[action]
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def init(args):
    action = args.action
    model_name = args.model
    # Point d'entrée unique : prépare la config puis délègue à l'action demandée.

    # Check environment
    # ht.check_requirement()
    # ht.check_python_version()
    # ht.check_tf_gpu_usable()

    # Create configuration file
    config(model_name=model_name)

    # check valid config
    cfg = validate_config()

    # check train dataset dir
    # ht.check_train_dataset_dir(cfg)

    # check train results dir
    # ht.check_train_results_dir(cfg)

    if action not in ACTIONS:
        print(f"Unknown action: {action}")
        print("Available actions: train | eval | validation | test")
        sys.exit(1)

    # Les options validation sont injectées dans cfg pour un passage uniforme.
    cfg["validation_config"] = {
        "image_path": args.validation_image,
        "threshold": args.validation_threshold,
        "model_path": args.validation_model,
        "dataset_dir": args.validation_dataset_dir,
        "show_plot": not args.validation_no_display,
    }

    # Résolution de l'action CLI vers la fonction métier correspondante (lazy import).
    func = _resolve_action_func(action)

    # On transmet la même configuration validée à tous les sous-modules.
    sig = inspect.signature(func)
    if len(sig.parameters) == 0:
        func()
    else:
        func(cfg)

def parse_args():
    parser = argparse.ArgumentParser(description="EfficientNet pipeline runner")

    parser.add_argument(
        "--action", "--actions",
        dest="action",
        required=True,
        choices=ACTIONS.keys(),
        help="What to do: train | eval | validation | test",
    )

    parser.add_argument(
        "--model",
        default="efficientnet-b0",
        help="Model name to use, e.g. efficientnet-b0",
    )

    parser.add_argument(
        "--validation-image",
        default=None,
        help="Image path used by validation action.",
    )
    parser.add_argument(
        "--validation-threshold",
        type=float,
        default=0.4,
        help="Confidence threshold used by validation action.",
    )
    parser.add_argument(
        "--validation-model",
        default=None,
        help="Model path (.h5/.keras) used by validation action.",
    )
    parser.add_argument(
        "--validation-dataset-dir",
        default=None,
        help="Dataset directory used to infer class order in validation action.",
    )
    parser.add_argument(
        "--validation-no-display",
        action="store_true",
        help="Do not open matplotlib window in validation action.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    init(args)
