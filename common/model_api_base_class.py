class ModelAPIBaseClass:
    def load(self, model_path: str):
        raise NotImplementedError()

    def save(self, save_path: str):
        raise NotImplementedError()

    def train(self, number_of_epochs: int, learning_rate: float):
        raise NotImplementedError()

    def generate_text(self, input: str):
        raise NotImplementedError()
