from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.Con = nn.Sequential(
            nn.Conv2d(kernel_size=3, stride=1, padding=1, in_channels=1, out_channels=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),


            nn.Conv2d(kernel_size=3, stride=1, padding=1, in_channels=4, out_channels=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        self.FC = nn.Sequential(
            nn.Linear(in_features=784, out_features=100),
            nn.LeakyReLU(),
            nn.Linear(in_features=100, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        y = self.Con(x)
        y = y.view(-1, y.shape[2]*y.shape[3])
        return self.FC(y)
