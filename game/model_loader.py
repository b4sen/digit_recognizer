import torch
from model.model import Net

class ModelLoader:

    def __init__(self, path: str):
        self.path = path 
        self.net = Net()

    def load_model(self):
        self.net.load_state_dict(torch.load(self.path, map_location=torch.device('cpu')))
        self.net.eval()
        return self.net