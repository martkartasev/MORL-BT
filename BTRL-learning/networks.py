import torch.nn as nn


class MLP_old(nn.Module):
    def __init__(self, input_size, output_size, hidden_arch=(16, 16, 16, 16), hidden_activation=nn.ELU, with_batchNorm=False, **kwargs):
        super().__init__()

        self.network = nn.Sequential()

        if with_batchNorm:
            self.network.add_module(
                "input",
                nn.Sequential(
                    nn.Linear(input_size, hidden_arch[0]),
                    nn.BatchNorm1d(hidden_arch[0]),
                    hidden_activation()
                )
            )
        else:
            self.network.add_module(
                "input",
                nn.Linear(input_size, hidden_arch[0])
            )

        for i in range(1, len(hidden_arch)):
            if with_batchNorm:
                self.network.add_module(
                    f"hidden_{i}",
                    nn.Sequential(
                        nn.Linear(hidden_arch[i-1], hidden_arch[i]),
                        nn.BatchNorm1d(hidden_arch[i]),
                        hidden_activation()
                    )
                )
            else:
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


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_arch=(16, 16, 16, 16), hidden_activation=nn.ELU, with_batchNorm=False, **kwargs):
        super().__init__()

        layers = []

        # input
        layers.append(nn.Linear(input_size, hidden_arch[0]))
        if with_batchNorm:
            layers.append(nn.BatchNorm1d(hidden_arch[0]))
        layers.append(hidden_activation())

        # hidden
        for i in range(1, len(hidden_arch)):
            layers.append(nn.Linear(hidden_arch[i - 1], hidden_arch[i]))
            if with_batchNorm:
                layers.append(nn.BatchNorm1d(hidden_arch[i]))
            layers.append(hidden_activation())

        # output
        layers.append(nn.Linear(hidden_arch[-1], output_size))

        self.network = nn.Sequential(*layers)

        # print(self.network)

    def forward(self, x):
        return self.network(x)

