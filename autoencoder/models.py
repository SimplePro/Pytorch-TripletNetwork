import torch.nn as nn
import torch.nn.functional as F
import torch


class Resize(nn.Module):

    def __init__(self, scale_factor=2.0, size=None):
        super().__init__()

        self.scale_factor = scale_factor
        self.size = size
    
    def forward(self, x):

        if self.size == None: return F.interpolate(x, scale_factor=self.scale_factor)
        else: return F.interpolate(x, size=self.size)


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2), # (32, 14, 14)
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2), # (64, 7, 7)
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2), # (128, 3, 3)
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1),
            Resize(size=(28, 28)),

            nn.Sigmoid(), # Sigmoid function makes reconstructed images not noisy
        )

    def get_rep(self, x):
        return self.encoder(x).view(-1, 128)

    def forward(self, x):

        code = self.encoder(x)
        return self.decoder(code)

if __name__ == '__main__':
    autoencoder = AutoEncoder() 
    x = torch.randn((2, 1, 28, 28))

    print(autoencoder(x))

    from torchsummary import summary

    summary(autoencoder.cuda(), input_size=(1, 28, 28))