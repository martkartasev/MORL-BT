import torch.nn as nn


class FCQNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            # nn.Tanh(),  # Q-function sometimes looks nicer with this
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.network(x)
