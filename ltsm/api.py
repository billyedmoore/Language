from common.model_api_base_class import ModelAPIBaseClass
from common.data_prep import prepare_input

from .model import LTSMmodel, generate_text, train
from rnn.dataset import RNNDataset
import torch

from pathlib import Path


class LTSMAPI(ModelAPIBaseClass):
    model: LTSMmodel | None = None
    dataset: RNNDataset | None = None
    device: torch.device

    def __init__(
        self,
        device: torch.device,
        input_length: int,
        input_filename: str,
        dataset_size: int,
        hidden_layer_size: int = 128,
    ):
        self.device = device
        self.dataset = None
        self.input_length = input_length
        self.input_filename = input_filename
        self.dataset_size = dataset_size
        self.hidden_layer_size = hidden_layer_size

    def _load_dataset(self):
        simple_filename = self.input_filename
        file_path = Path(__file__).resolve().parent.parent
        data_folder = file_path.joinpath("data")
        input_file = data_folder.joinpath(f"{simple_filename}.txt")
        prepared_input = data_folder.joinpath(f"prepared_{simple_filename}.txt")

        if not prepared_input.exists():
            prepare_input(input_file, prepared_input)
        self.dataset = RNNDataset(
            prepared_input,
            self.dataset_size,
            self.input_length,
            self.device,
            random_seed=913,
        )

    def _create_model(self):
        if self.dataset is None:
            self._load_dataset()

        assert self.dataset is not None

        self.model = LTSMmodel(len(self.dataset.i_to_char), self.hidden_layer_size)
        self.model.to(self.device)

    def load(self, model_path: str):
        self._create_model()

        assert self.model != None

        self.model.load_state_dict(torch.load(model_path))

    def save(self, save_path: str):
        if self.model is None:
            raise ValueError("No Model Loaded")

        torch.save(self.model.state_dict(), save_path)

    def train(
        self, number_of_epochs: int, learning_rate: float, early_stopping: bool = True
    ):

        if self.dataset is None:
            self._load_dataset()

        assert self.dataset is not None

        self._create_model()

        assert self.model is not None

        train(
            self.model,
            self.dataset,
            self.device,
            num_epochs=number_of_epochs,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
        )

    def generate_text(self, input: str):
        if self.dataset is None:
            self._load_dataset()

        assert self.dataset != None

        if self.model is None:
            raise ValueError("No Model Loaded")

        return generate_text(self.model, input, 30, self.dataset, self.device)
