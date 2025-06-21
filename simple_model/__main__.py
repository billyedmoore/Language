from .dataset import SimpleModelDataset
from .model import SimpleModel, train
from .model_utils import predict_next_char 
from data_prep import prepare_input
import datetime
import os
import torch



def main(input_length):
    simple_filename = "shakespeare"
    input_file = f"../data/{simple_filename}.txt"
    prepared_input = f"../data/prepared_{simple_filename}.txt"
    print(datetime.datetime.now())

    if not os.path.exists(prepared_input):
        prepare_input(input_file,prepared_input)


    train_data = SimpleModelDataset(prepared_input, 10000, input_length, random_seed=913
    )
    eval_data = SimpleModelDataset(prepared_input, 1000, input_length, random_seed=340
    )


    model = SimpleModel(input_length,len(train_data.i_to_char))
    print(datetime.datetime.now())
    train(model, train_data, eval_data, num_epochs=100)
    print(datetime.datetime.now())

    start = "to be or "
    while len(start) < 20:
        start += predict_next_char(model,eval_data,start)
        print(repr(start))


if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU.")
    main(200)
