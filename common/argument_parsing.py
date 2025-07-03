import argparse

from .model_api_base_class import ModelAPIBaseClass


def execute_by_arguments(args: argparse.Namespace, model: ModelAPIBaseClass):
    """
    Exectute from parser arguments (from the parser returned by `get_argparser`
    """
    match args.command:
        case "train":
            model.train(args.num_epochs, args.learning_rate)
            model.save(args.save_path)
        case "generate":
            model.load(args.model_path)
            model.generate_text(args.input_text)
        case _:
            raise ValueError("Unknown command")


def get_argparser(model_description: str):
    """
    Get the parser for the standard model train and generate API.
    """
    parser = argparse.ArgumentParser(description=model_description)

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")

    train_parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        dest="num_epochs",
        help="Required: Number of training epochs.",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        required=True,
        dest="learning_rate",
        help="Required: Learning rate for training.",
    )

    train_parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="save_path",
        help="Optional: Output file path for the trained model, by default model is not saved.",
    )

    generate_parser = subparsers.add_parser("generate")

    generate_parser.add_argument(
        "--input-text",
        type=str,
        required=True,
        dest="input_text",
        help="Required: Input text for generation.",
    )

    generate_parser.add_argument(
        "--model",
        type=str,
        required=True,
        dest="model_path",
        help="Required: Path to the saved model.",
    )

    return parser
