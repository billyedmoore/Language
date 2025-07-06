import torch
import io 
from rnn.dataset import RNNDataset
from common.early_stopping import stop_early_naive
import random


class LTSMmodel(torch.nn.Module):
    def __init__(self, numb_categories: int, hidden_size: int):
        super(LTSMmodel, self).__init__()
        self.number_lstm_layers = 1
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(numb_categories, hidden_size, 
                                  self.number_lstm_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, numb_categories)

    def forward(self, x, hidden_state, cell_state):
        x, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))
        x = self.linear(x.reshape(-1, self.hidden_size))
        return x, (hidden_state, cell_state)

    def create_state_layer(self, batch_size: int, device: torch.device):
        return torch.zeros(self.number_lstm_layers,
                           batch_size, self.hidden_size, device=device)


def train(
    model: LTSMmodel,
    dataset: RNNDataset,
    device: torch.device,
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

    train_losses = []
    eval_losses = []
    checkpoints = []

    for epoch in range(num_epochs):
        train_loss = 0
        for _, (inputs, targets) in enumerate(train_loader):
            hidden = model.create_state_layer(inputs.size(0), device)
            cell = model.create_state_layer(inputs.size(0), device)
            optimizer.zero_grad()
            outputs, (hidden, cell) = model(inputs, hidden, cell)
            loss = criterion(outputs, targets.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(eval_loader):
                hidden = model.create_state_layer(inputs.size(0), device)
                cell = model.create_state_layer(inputs.size(0), device)
                outputs, (hidden, cell) = model(inputs, hidden, cell)
                loss = criterion(outputs, targets.view(-1))
                eval_loss += loss.item() * inputs.size(0)
        model.train()

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        torch.save(model, "temp") 
        checkpoints.append("temp")


        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Validation Loss: {eval_loss / len(eval_loader):.5f}, "
        )

        if stop_early_naive(eval_losses):
            print(f"Stopping Early at Epoch {epoch+1}")
            i = eval_losses.index(max(eval_losses))
            from_checkpoint = torch.load(checkpoints[i],weights_only=False)
            model.load_state_dict(from_checkpoint.state_dict())
            print(f"Restoring To Epoch {epoch+1}")
            
            break


def generate_text(
    model: LTSMmodel,
    input_str: str,
    length_to_add: int,
    dataset: RNNDataset,
    device: torch.device,
):
    model.eval()
    hidden = model.create_state_layer(1, device)
    cell = model.create_state_layer(1, device)

    generated_text = input_str

    for c in input_str:
        input = torch.unsqueeze(dataset.encode_input(c), dim=0)
        _, (hidden, cell) = model(input, hidden, cell)

    previous_char = input_str[-1]

    for _ in range(length_to_add):
        input = torch.unsqueeze(dataset.encode_input(previous_char), dim=0)
        output, (hidden, cell) = model(input, hidden, cell)

        probabilites = torch.softmax(output, dim=1).tolist()[0]
        selected = random.choices(
            [i for i in range(len(probabilites))], weights=probabilites, k=1
        )
        previous_char = dataset.i_to_char[selected[0]]
        generated_text += previous_char

    model.train()
    return generated_text
