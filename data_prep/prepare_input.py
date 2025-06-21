import os
import re
from collections import Counter
from simple_model.model_utils import get_frequencies
from pathlib import Path

CHUNK_SIZE = 1024


def prepare_input(input_filename: Path, output_filename: Path, minimum_freq: int = 100):
    """
    Remove infrequent characters from file {input_filename} and save result to {output_filename}.
    """
    if not input_filename.exists():
        raise FileNotFoundError(f"Input file {input_filename} does not exist.")

    char_freq = get_frequencies(input_filename)

    chars_to_remove = [
        char for (char, freq) in char_freq.items() if freq < minimum_freq
    ]

    with open(output_filename, "w") as f:
        f.write("")

    with open(input_filename, "r") as input_f:
        while 1:
            chunk = input_f.read(CHUNK_SIZE)
            chunk = re.sub("“|”", '"', chunk)
            chunk = re.sub("’|‘", "'", chunk)
            chunk = re.sub("\n+", "\n", chunk)
            chunk = re.sub(" +", " ", chunk)

            chunk = chunk.encode("ascii", "ignore").decode("ascii")
            # If nothing left to read
            if not chunk:
                break

            with open(output_filename, "a") as output_f:
                output_f.write(
                    "".join([c for c in chunk.lower() if c not in chars_to_remove])
                )

    print(_get_frequencies(output_filename))
