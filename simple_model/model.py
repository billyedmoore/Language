import torch
import math
from .model_utils import get_frequencies
from .dataset import SimpleModelDataset


class SimpleModel(torch.nn.Module):
    def __init__(self, input_length: int, numb_categories: int):
        super(SimpleModel, self).__init__()
        input_size = input_length * numb_categories
        self.linear1 = torch.nn.Linear(input_size, input_size // 2)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(input_size // 2, numb_categories)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return x


def _char_frequency_to_class_weights(
    char_freq: dict[str, int], char_to_i: dict[str, int]
) -> torch.Tensor:
    n_chars = sum(char_freq.values())
    weights_dict = {}

    if set(list(char_freq.keys()) + [""]) != set(char_to_i.keys()):
        raise ValueError(
            "frequency table and character tokens should have the same keys"
        )

    for c, freq in char_freq.items():
        if freq == 0:
            raise ValueError("No item should have frequency 0")
        else:
            # the more frequent the lower the weight
            # weight = 1/(freq ** 0.5) * 10
            weight = 1 / math.log(freq + 10)
            weights_dict[c] = weight

    weights = [-1 for _ in range(len(char_to_i))]
    weights[0] = 0
    for c, weight in weights_dict.items():
        i = char_to_i[c]
        weights[i] = weight
    print(weights)
    return torch.tensor(weights, dtype=torch.float32)


def train(
    model: torch.nn.Module,
    dataset: SimpleModelDataset,
    device: torch.device,
    num_epochs: int = 1000,
):
    print(device)
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [0.8, 0.2], generator=torch.Generator(device=device)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, generator=torch.Generator(device=device)
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=16,
        shuffle=True,
        generator=torch.Generator(device=device),
    )

    # train_weights = _char_frequency_to_class_weights(
    #    get_frequencies(train_dataset.file_name), train_dataset.char_to_i
    # )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    for epoch in range(num_epochs):
        train_loss = 0
        for _, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for inputs, targets in eval_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                eval_loss += loss.item() * inputs.size(0)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Validation Loss: {eval_loss / len(eval_loader):.4f}"
        )
