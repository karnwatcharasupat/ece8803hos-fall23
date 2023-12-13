from torch import nn

class TwoLayerReLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bias_first=False, bias_second=False):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias_first)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias_second)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x