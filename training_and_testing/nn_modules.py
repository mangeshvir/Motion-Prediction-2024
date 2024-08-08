import torch
import torch.nn as nn
import torch.nn.functional as F

class ConstantVelocityModel(nn.Module):
    def __init__(self, dt=1.0):
        """
        Constant Velocity Model.
        """
        super(ConstantVelocityModel, self).__init__()
        self.dt = dt

    def forward(self, x):
        """
        Forward pass.
        """
        x = x[:, -1, :]
        return x + self.dt * x

    def loss_function(self, y_hat, y):
        """
        Mean Squared Error loss.
        """
        return F.mse_loss(y_hat, y)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Multi-Layer Perceptron Model.
        """
        super(MultiLayerPerceptron, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass.
        """
        batch_size = x.shape[0]
        x = x.flatten(start_dim=1)
        x = self.layers(x)
        return x.view(batch_size, -1, self.output_dim)

    def loss_function(self, y_hat, y):
        """
        Mean Squared Error loss.
        """
        return F.mse_loss(y_hat, y)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, future_sequence_length):
        """
        LSTM Model.
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.future_sequence_length = future_sequence_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * future_sequence_length)

    def forward(self, x):
        """
        Forward pass.
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.view(-1, self.future_sequence_length, out.size(-1) // self.future_sequence_length)

    def loss_function(self, y_hat, y):
        """
        Mean Squared Error loss.
        """
        return F.mse_loss(y_hat, y)
