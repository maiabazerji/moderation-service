from train.train import run as train_run
from validation.validation import run as eval_run
from validation.validation_tflite import run as eval_tflite_run
from test.test import run as test_run
import tools.hardware_test as ht
from tools.configuration_generator import config
import sys
import argparse
from tools.config_validator import validate_config

ACTIONS = {
    "train": train_run,
    "eval": eval_run,
    "eval_tflite": eval_tflite_run,
    "test": test_run,
}

def init(action: str, model_name: str):
    ht.check_requirement()
    ht.check_python_version()
    ht.check_gpus()

    config(model_name=model_name)

    cfg = validate_config()

    ht.check_train_dataset_dir(cfg)
    ht.check_train_results_dir(cfg)

    func = ACTIONS[action]
    func(cfg)

def parse_args():
    parser = argparse.ArgumentParser(description="EfficientNet pipeline runner")

    parser.add_argument(
        "--action", "--actions",
        dest="action",
        required=True,
        choices=ACTIONS.keys(),
        help="What to do: train | eval | eval_tflite | test",
    )

    parser.add_argument(
        "--model",
        default="efficientnet-b0",
        help="Model name to use, e.g. efficientnet-b0",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    init(args.action, args.model)
