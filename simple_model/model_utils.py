import random
from collections import Counter
from pathlib import Path
import torch


def get_frequencies(input_filename: Path) -> dict[str, int]:
    char_freq = Counter()

    with open(input_filename, "r") as f:
        while 1:
            chunk = f.read(1024)

            # If nothing left to read
            if not chunk:
                break
            char_freq.update(chunk.lower())
    return dict(char_freq)


def predict_next_char(model, dataset, input: str):
    input_tensor: torch.Tensor = dataset.encode_input(input).unsqueeze(0)
    probabilites = torch.softmax(model(input_tensor), dim=1).tolist()[0]

    print(probabilites)

    char = ""
    # If we are predicting a placeholder
    while char == "":
        selected = random.choices(
            [i for i in range(len(probabilites))], weights=probabilites, k=1
        )
        char = dataset.i_to_char[selected[0]]

    print(char)
    return char
