from torch import nn

# Model definition


class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            # 3 input channels, 16 output channels
            nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2),
            # Normalize the batch
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(3*3*64, 10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 2)
        self.relu = nn.ReLU()

    def forward(self, x):

        out = self.layer1(x)

        out = self.layer2(out)

        out = self.layer3(out)

        out = out.view(out.size(0), -1)

        out = self.relu(self.fc1(out))

        out = self.fc2(out)

        return out
