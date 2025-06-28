import torch
from .dataset import RNNDataset
import random


class RNNModel(torch.nn.Module):
    def __init__(self, numb_categories: int, hidden_size: int):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(numb_categories, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, numb_categories)

    def forward(self, x, hidden_state):
        x, hidden_state = self.rnn(x, hidden_state)
        x = self.linear(x.reshape(-1, self.hidden_size))
        return x, hidden_state

    def create_hidden(self, batch_size: int):
        return torch.zeros(1, batch_size, self.hidden_size)


def train(
    model: RNNModel,
    dataset: RNNDataset,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
):
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=16,
        shuffle=True,
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = 0
        for _, (inputs, targets) in enumerate(train_loader):
            hidden = model.create_hidden(inputs.size(0))
            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(eval_loader):
                hidden = model.create_hidden(inputs.size(0))
                outputs, hidden = model(inputs, hidden)
                loss = criterion(outputs, targets.view(-1))
                eval_loss += loss.item() * inputs.size(0)
        model.train()
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Validation Loss: {eval_loss / len(eval_loader):.5f}, "
            f"Learning Rate: {optimizer.param_groups[0]['lr']}"
        )


def generate_text(
    model: RNNModel, input_str: str, length_to_add: int, dataset: RNNDataset
):
    model.eval()
    hidden = model.create_hidden(1)

    generated_text = input_str

    for c in input_str:
        input = torch.unsqueeze(dataset.encode_input(c), dim=0)
        _, hidden = model(input, hidden)

    previous_char = input_str[-1]

    for _ in range(length_to_add):
        input = torch.unsqueeze(dataset.encode_input(previous_char), dim=0)
        output, hidden = model(input, hidden)

        probabilites = torch.softmax(output, dim=1).tolist()[0]
        selected = random.choices(
            [i for i in range(len(probabilites))], weights=probabilites, k=1
        )
        previous_char = dataset.i_to_char[selected[0]]
        generated_text += previous_char

    model.train()
    return generated_text
