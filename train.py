import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

import torchvision.transforms as TF
from torchvision.datasets import MNIST

from models import TripletNetwork
from utils import TripletDataset

import matplotlib.pyplot as plt


DEVICE = "cuda"
EPOCHS = 30
LEARNING_RATE = 0.5
LEARNING_RATE_DECAY = 0.5
MOMENTUM = 0.9
CONST = 2


if __name__ == '__main__':

    model = TripletNetwork(model="mnist").to(DEVICE)
    
    transform = TF.Compose([
        TF.ToTensor(),
        TF.Normalize([0], [1])
    ])

    train_mnist = MNIST(root="./mnist", train=True, download=True, transform=transform)
    valid_mnist = MNIST(root="./mnist", train=False, download=True, transform=transform)

    trainset = TripletDataset(
        dataset=train_mnist,
        class_n=10,
        transform=TF.Compose([])
    )

    validset = TripletDataset(
        dataset=valid_mnist,
        class_n=10,
        transform=TF.Compose([])
    )

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    validloader = DataLoader(validset, batch_size=128)

    opt = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=LEARNING_RATE_DECAY, verbose=False)

    train_history = []
    valid_history = []

    def calculate_loss(pos_distance, neg_distance):
        d_pos = torch.exp(pos_distance) / (torch.exp(pos_distance) + torch.exp(neg_distance))
        d_neg = torch.exp(neg_distance) / (torch.exp(pos_distance) + torch.exp(neg_distance))

        # print(d_pos, d_neg)

        loss = torch.mean(torch.square(d_pos) + torch.square(d_neg - 1))

        return loss

    for epoch in range(EPOCHS):
        print("-" * 50 + "EPOCH:", epoch, "-" * 50)

        model.train()
        for (x, _, x_pos, _, x_neg, _) in trainloader:
            x, x_pos, x_neg = x.to(DEVICE), x_pos.to(DEVICE), x_neg.to(DEVICE)

            pos_distance, neg_distance = model(x, x_pos, x_neg)
            # print(torch.mean(pos_distance), torch.mean(neg_distance))
            loss = calculate_loss(pos_distance, neg_distance)
            # print(loss)

            train_history.append(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

        lr_scheduler.step()

        # model.eval() # affects Dropout
        with torch.no_grad():
            for (x, _, x_pos, _, x_neg, _) in validloader:
                x, x_pos, x_neg = x.to(DEVICE), x_pos.to(DEVICE), x_neg.to(DEVICE)

                # print(torch.mean(x), torch.mean(x_pos), torch.mean(x_neg))
                pos_distance, neg_distance = model(x, x_pos, x_neg)
                # print(torch.mean(pos_distance), torch.mean(neg_distance))

                loss = calculate_loss(pos_distance, neg_distance)
                # print(loss)

                valid_history.append(loss.item())

        plt.plot(range(len(train_history)), train_history, label="train_loss", color="blue")
        plt.legend()
        plt.tick_params(axis='x', labelcolor='blue')
        
        ax = plt.twiny()
        ax.plot(range(len(valid_history)), valid_history, label="valid_loss", color="red")
        ax.legend()
        plt.tick_params(axis="x", labelcolor="red")

        # plt.legend()
        plt.savefig("./loss_history.jpg")
        # plt.show()

        plt.clf()