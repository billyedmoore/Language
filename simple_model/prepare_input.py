import os
import re
from collections import Counter

CHUNK_SIZE = 1024 

def _get_frequencies(input_filename: str) -> dict[str,int]:
    char_freq = Counter()

    with open(input_filename, "r") as f:
        while 1:
            chunk = f.read(CHUNK_SIZE)

            # If nothing left to read
            if not chunk:
                break
            char_freq.update(chunk.lower())
    print(char_freq)
    return dict(char_freq)

def prepare_input(input_filename: str, output_filename: str, minimum_freq: int = 100):
    """ 
    Remove infrequent characters from file {input_filename} and save result to {output_filename}.
    """
    if not os.path.exists(input_filename):
        raise FileNotFoundError(f"Input file {input_filename} does not exist.")

    char_freq = _get_frequencies(input_filename)

    chars_to_remove = [char for (char,freq) in char_freq.items() if freq < minimum_freq]
    
    with open(output_filename, "w") as f:
        f.write("")
    
    with open(input_filename, "r") as input_f:
        while 1:
            chunk = input_f.read(CHUNK_SIZE)
            chunk = re.sub("\n+","\n",chunk)
            chunk = re.sub(" +"," ",chunk)
            # If nothing left to read
            if not chunk:
                break

            with open(output_filename, "a") as output_f:
                output_f.write("".join([c for c in chunk.lower() if c not in chars_to_remove]))





