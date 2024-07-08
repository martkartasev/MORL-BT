import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32, squash_output=False, hidden_activation=nn.ELU):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            hidden_activation(),
            nn.Linear(hidden_size, hidden_size),
            hidden_activation(),
            nn.Linear(hidden_size, hidden_size),
            hidden_activation(),
            # nn.Tanh(),  # Q-function sometimes looks nicer with this
            nn.Linear(hidden_size, output_size),
        )

        if squash_output:
            self.network = self.network.append(nn.Sigmoid())

        print(self.network)

    def forward(self, x):
        return self.network(x)
