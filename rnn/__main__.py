from .dataset import RNNDataset
from .model import RNNModel, train, generate_text

# from .model_utils import predict_next_char
from data_prep import prepare_input
from pathlib import Path
import torch


def main(input_length):
    simple_filename = "shakespeare"
    file_path = Path(__file__).resolve().parent.parent
    data_folder = file_path.joinpath("data")
    input_file = data_folder.joinpath(f"{simple_filename}.txt")
    prepared_input = data_folder.joinpath(f"prepared_{simple_filename}.txt")

    if not prepared_input.exists():
        prepare_input(input_file, prepared_input)

    dataset = RNNDataset(prepared_input, 10000, input_length, random_seed=913)

    model = RNNModel(len(dataset.i_to_char), 128)
    train(model, dataset, num_epochs=100, learning_rate=0.01)

    start = "to be or "
    print(repr(generate_text(model, start, 30, dataset)))


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU.")
    main(200)
