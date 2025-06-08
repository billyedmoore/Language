from dataset import SimpleModelDataset
from model import SimpleModel, train


def main(input_length):
    train_data = SimpleModelDataset(
        "../data/wordsworth.txt", 100000, input_length, random_seed=913
    )
    eval_data = SimpleModelDataset(
        "../data/wordsworth.txt", 1000, input_length, random_seed=340
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
