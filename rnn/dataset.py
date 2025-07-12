import torch
import random
import os
from pathlib import Path


class RNNDataset(torch.utils.data.Dataset):
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
    
        first_offset = self.random.randint(0,self.input_length-1)
        max_start_index = self.file_length-self.input_length-first_offset

        self.start_indexes = list(range(first_offset, max_start_index, self.input_length))

        max_n_samples = len(self.start_indexes)

        if self.number_of_samples <= max_n_samples:
            print(self.number_of_samples, max_n_samples)
            self.start_indexes = self.random.sample(self.start_indexes,self.number_of_samples)
        else:
            raise ValueError(f"Text too short to support {self.number_of_samples} samples, can only provide {max_n_samples}")


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

        char_dict = {c: i for (i, c) in enumerate(unique_chars)}

        self.char_to_i = char_dict
        char_list: list[str] = ["TO_BE_POPULATED"] * (max(char_dict.values()) + 1)

        for c, i in char_dict.items():
            char_list[i] = c

        assert "TO_BE_POPULATED" not in char_list

        self.i_to_char = char_list

    def encode_input(self, input: str) -> torch.Tensor:
        input_indexes = [self.char_to_i[c] for c in input]
        index_tensor = torch.tensor(
            input_indexes, dtype=torch.int64, device=self.device
        )
        input_tensor = torch.zeros(
            self.input_length, len(self.i_to_char), device=self.device
        )
        input_tensor = input_tensor.scatter(1, index_tensor.unsqueeze(1), 1)
        return input_tensor

    def _encode_target(self, target: str) -> torch.Tensor:
        raw_label_tensor = torch.tensor(
            [self.char_to_i[c] for c in target], dtype=torch.int64, device=self.device
        )
        return raw_label_tensor

    def __getitem__(self, idx):
        """
        Get a single (x,y) pair by id.
        """
        char_index = self.start_indexes[idx]

        assert char_index >= 0 and char_index < self.file_length

        with open(self.file_name, "r") as f:
            f.seek(char_index)
            seq = f.read(self.input_length + 1)

        input_tensor = self.encode_input(seq[:-1])
        label_tensor = self._encode_target(seq[1:])

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
