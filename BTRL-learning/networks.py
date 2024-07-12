import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_arch=(16, 16, 16, 16), hidden_activation=nn.ELU, **kwargs):
        super().__init__()

        self.network = nn.Sequential()
        self.network.add_module(
            "input",
            nn.Linear(input_size, hidden_arch[0])
        )

        for i in range(1, len(hidden_arch)):
            self.network.add_module(
                f"hidden_{i}",
                nn.Sequential(
                    nn.Linear(hidden_arch[i-1], hidden_arch[i]),
                    hidden_activation()
                )
            )

        self.network.add_module(
            "output",
            nn.Linear(hidden_arch[-1], output_size)
        )

        print(self.network)

    def forward(self, x):
        return self.network(x)
