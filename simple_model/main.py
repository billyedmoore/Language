from dataset import SimpleModelDataset
from model import SimpleModel, train
from prepare_input import prepare_input
import os 


def main(input_length):
    input_file = "../data/wordsworth.txt"
    prepared_input = "../data/prepared_wordsworth.txt"

    #if not os.path.exists(prepared_input):
    prepare_input(input_file,prepared_input)


    train_data = SimpleModelDataset(prepared_input, 100000, input_length, random_seed=913
    )
    eval_data = SimpleModelDataset(prepared_input, 1000, input_length, random_seed=340
    )


    model = SimpleModel(input_length)
    train(model, train_data, eval_data)

    for i in range(0, 100):
        (context, label) = eval_data[i]
        print(
            f"{repr(eval_data.input_to_str(context))} -> {eval_data.output_to_char(label)} vs predicted {eval_data.output_to_char(model(context))}"
        )


if __name__ == "__main__":
    main(30)
