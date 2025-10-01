# src/NeuralNetwork.py
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x) # apply sigmoid activation to output layer
        return x