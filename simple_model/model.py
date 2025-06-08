import torch


class SimpleModel(torch.nn.Module):
    def __init__(self, input_length: int):
        super(SimpleModel, self).__init__()
        self.linear1 = torch.nn.Linear(input_length, 512)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(512, 256)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


def train(
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    eval_dataset: torch.utils.data.Dataset,
    num_epochs: int = 10000,
):
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

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        train_loss = 0
        for batch, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
