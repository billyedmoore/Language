import torch
import random
import os
from pathlib import Path


class SimpleModelDataset(torch.utils.data.Dataset):
    i_to_char: list[str]
    char_to_i: dict[str, int]

    def __init__(
        self,
        file_name: Path,
        number_of_samples: int,
        input_length: int,
        device: torch.device,
        random_seed: int | None = None,
    ):
        self.file_name = file_name
        self._create_character_dictionary()
        self.random = random.Random(random_seed)
        self.number_of_samples = number_of_samples
        self.file_length = os.path.getsize(file_name)
        self.input_length = input_length
        self.device = device

        self.sample_input_length = [
            self.random.randint(2, self.input_length) for _ in range(number_of_samples)
        ]
        self.sample_predict_char_index = [
            self.random.randint(3, self.file_length) for _ in range(number_of_samples)
        ]

    def _create_character_dictionary(self):
        """
        Create `i_to_char` and `char_to_i` using all chars included in the corpus.
        """
        unique_chars = set()
        with open(self.file_name, "r") as f:
            while 1:
                chunk = f.read(1024)

                # If nothing left to read
                if not chunk:
                    break

                unique_chars.update(chunk)

        char_dict = {c: i + 1 for (i, c) in enumerate(unique_chars)}
        char_dict[""] = 0

        self.char_to_i = char_dict
        char_list: list[str] = ["TO_BE_POPULATED"] * (max(char_dict.values()) + 1)

        for c, i in char_dict.items():
            char_list[i] = c

        assert "TO_BE_POPULATED" not in char_list

        self.i_to_char = char_list

    def encode_input(self, input: str) -> torch.Tensor:
        pad_length = self.input_length - len(input)
        input_indexes = [0 for _ in range(pad_length)] + [
            self.char_to_i[c] for c in input
        ]
        index_tensor = torch.tensor(
            input_indexes, dtype=torch.int64, device=self.device
        )
        input_tensor = torch.zeros(
            self.input_length, len(self.i_to_char), device=self.device
        )
        input_tensor = input_tensor.scatter(1, index_tensor.unsqueeze(0), 1)
        return input_tensor.flatten()

    def encode_target(self, target: str) -> torch.Tensor:
        assert len(target) == 1
        raw_label_tensor = torch.tensor(
            self.char_to_i[target], dtype=torch.int64, device=self.device
        )
        return raw_label_tensor

    def __getitem__(self, idx):
        """
        Get a single (x,y) pair by id.
        """
        given_input_length = self.sample_input_length[idx]
        char_index = self.sample_predict_char_index[idx]

        assert char_index >= 0 and char_index < self.file_length

        if (char_index - given_input_length) < 0:
            given_input_length = char_index

        first_i = char_index - given_input_length

        with open(self.file_name, "r") as f:
            f.seek(first_i)
            input_str = f.read(given_input_length)
            label = f.read(1)

        input_tensor = self.encode_input(input_str)
        label_tensor = self.encode_target(label)

        return input_tensor, label_tensor

    def __len__(self):
        return self.number_of_samples

    def output_to_char(self, output: torch.Tensor) -> str:
        return self.i_to_char[int(output.cpu().tolist()[0])]

    def input_to_str(self, input: torch.Tensor) -> str:
        string = ""
        for i in input.cpu().tolist():
            string += self.i_to_char[int(i)]
        return string
