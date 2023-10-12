import torch.nn as nn
import torch.nn.functional as F
import torch


MODEL_CONFIGURATIONS = {
    "mnist": {
        "filter": [5, 3, 2, 2],
        "feature_dim": [1, 32, 64, 128, 128],
        "stride": [1, 1, 1, 1],
        "padding": [0, 0, 0, 0]
    }
}


class TripletNetwork(nn.Module):

    def __init__(self, model="mnist"):
        super().__init__()

        self.model = model
        model_configuration = MODEL_CONFIGURATIONS[self.model]        

        self.main = []

        for i in range(4):
            self.main.append(
                nn.Conv2d(in_channels=model_configuration["feature_dim"][i], out_channels=model_configuration["feature_dim"][i+1],
                        kernel_size=model_configuration["filter"][i], stride=model_configuration["stride"][i],
                        padding=model_configuration["padding"][i]),
            )

            if i != 3:
                self.main.append(nn.MaxPool2d(kernel_size=2, stride=2))
                self.main.append(nn.ReLU())

        self.main.append(nn.Dropout(0.2)) # difference from the reference's setting (0.5)

        self.main = nn.Sequential(*self.main)

    def get_rep(self, x):
        return self.main(x).view(-1, 128)
    
    def forward(self, x, pos_x, neg_x):
        x_rep = self.get_rep(x)
        pos_x_rep = self.get_rep(pos_x)
        neg_x_rep = self.get_rep(neg_x)

        pos_distance = torch.norm(x_rep - pos_x_rep, p=2, dim=1)
        neg_distance = torch.norm(x_rep - neg_x_rep, p=2, dim=1)

        return pos_distance, neg_distance


if __name__ == '__main__':
    triplet_network = TripletNetwork(model="mnist") 
    x = torch.randn((2, 1, 28, 28))
    pos_x = torch.randn((2, 1, 28, 28))
    neg_x = torch.randn((2, 1, 28, 28))

    print(triplet_network(x, pos_x, neg_x))

    from torchsummary import summary

    summary(triplet_network.cuda(), input_size=[(1, 28, 28), (1, 28, 28), (1, 28, 28)])