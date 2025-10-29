import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4,4)), #par defaut conv:  stride = 1, padding=0
                            nn.ReLU(), 
                            nn.MaxPool2d(kernel_size=3),
                            
                            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4,4)), 
                            nn.ReLU(), 
                            nn.MaxPool2d(kernel_size=3),

                            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4,4)),
                            nn.ReLU(), 
                            nn.MaxPool2d(kernel_size=3),

                            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,4)),
                            nn.ReLU()
                            
        )

        # to know the outputs size before FC layer
        with torch.no_grad():
            tmp = torch.zeros(1, 1, 150, 150) # size of the input
            out = self.conv(tmp)
            out = torch.flatten(out, start_dim=1, end_dim=3)
            n_features = out.shape[-1]

        self.fc = nn.Sequential(
                            nn.Flatten(start_dim=1, end_dim=3),
                            nn.Linear(in_features=n_features, out_features=4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x




