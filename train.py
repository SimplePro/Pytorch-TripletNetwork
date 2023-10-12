import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

import torchvision.transforms as TF
from torchvision.datasets import MNIST

from models import TripletNetwork
from utils import TripletDataset
import visualization_embedding_space as visualize_

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


DEVICE = "cuda"
EPOCHS = 30
LEARNING_RATE = 0.02 # difference from the reference's setting (0.5)
LEARNING_RATE_DECAY = 0.9
MOMENTUM = 0.9
CONST = 2


def calculate_loss(pos_distance, neg_distance):
        d_pos = torch.exp(pos_distance) / (torch.exp(pos_distance) + torch.exp(neg_distance))
        # d_neg = torch.exp(neg_distance) / (torch.exp(pos_distance) + torch.exp(neg_distance))
        # print(d_pos, d_neg)

        # loss = torch.mean(torch.square(d_pos) + torch.square(d_neg - 1))
        loss = CONST * torch.mean(torch.square(d_pos))

        return loss
    
    
@torch.no_grad()
def save_representation_space(model, dataset, class_n, save_path):
    class_embedding = visualize_.get_embedding(dataset=dataset, class_n=class_n, model=model)
    visualize_.save_2d_embedding_space(class_embedding, save_path)


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

    best_loss = 1e8
    best_state_dict = None

    def calculate_loss(pos_distance, neg_distance):
        d_pos = torch.exp(pos_distance) / (torch.exp(pos_distance) + torch.exp(neg_distance))
        # d_neg = torch.exp(neg_distance) / (torch.exp(pos_distance) + torch.exp(neg_distance))

        # print(d_pos, d_neg)

        # loss = torch.mean(torch.square(d_pos) + torch.square(d_neg - 1))
        loss = CONST * torch.mean(torch.square(d_pos))

        return loss
    
    
    @torch.no_grad()
    def save_representation_space(model, dataset, class_n, save_path):
        class_embedding = visualize_.get_embedding(dataset=dataset, class_n=class_n, model=model)
        visualize_.save_2d_embedding_space(class_embedding, save_path)

        
    for epoch in range(EPOCHS):
        print("-" * 50 + "EPOCH:", epoch, "-" * 50)
        train_avg_loss = 0
        valid_avg_loss = 0
        pos_distance_mean = 0
        neg_distance_mean = 0

        model.train()
        for (x, _, x_pos, _, x_neg, _) in trainloader:
            x, x_pos, x_neg = x.to(DEVICE), x_pos.to(DEVICE), x_neg.to(DEVICE)
            # print(y, y_pos, y_neg)
            # TF.ToPILImage()(x[0].cpu()).save("x.jpg")
            # TF.ToPILImage()(x_pos[0].cpu()).save("x_pos.jpg")
            # TF.ToPILImage()(x_neg[0].cpu()).save("x_neg.jpg")

            pos_distance, neg_distance = model(x, x_pos, x_neg)
            # print(torch.mean(pos_distance), torch.mean(neg_distance))
            loss = calculate_loss(pos_distance, neg_distance)
            # print(loss)

            train_history.append(loss.item())
            train_avg_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

        lr_scheduler.step()
        train_avg_loss /= len(trainloader)

        # model.eval() # affects Dropout
        with torch.no_grad():
            for (x, _, x_pos, _, x_neg, _) in validloader:
                x, x_pos, x_neg = x.to(DEVICE), x_pos.to(DEVICE), x_neg.to(DEVICE)

                # print(torch.mean(x), torch.mean(x_pos), torch.mean(x_neg))
                pos_distance, neg_distance = model(x, x_pos, x_neg)

                pos_distance_mean += torch.mean(pos_distance).item()
                neg_distance_mean += torch.mean(neg_distance).item()

                # print(torch.mean(pos_distance), torch.mean(neg_distance))

                loss = calculate_loss(pos_distance, neg_distance)
                # print(loss)

                valid_history.append(loss.item())
                valid_avg_loss += loss.item()

            valid_avg_loss /= len(validloader)
            pos_distance_mean /= len(validloader)
            neg_distance_mean /= len(validloader)

            if valid_avg_loss <= best_loss:
                best_loss = valid_avg_loss
                best_state_dict = model.state_dict()

                torch.save(best_state_dict, "./triplet_best_state_dict.pt")
                
        figure, ax1 = plt.subplots()

        plt.title("Loss History (blue: train, red: valid)")

        ax1.plot(range(len(train_history)), train_history, label="train_loss", color="blue")
        ax1.tick_params(axis='x', labelcolor='blue')
        
        ax2 = ax1.twiny()
        ax2.plot(range(len(valid_history)), valid_history, label="valid_loss", color="red")
        ax2.tick_params(axis="x", labelcolor="red")

        plt.savefig("./loss_history.jpg")
        plt.cla()

        save_representation_space(model=model, dataset=valid_mnist, class_n=10, save_path=f"./embedding_space/triplet/epoch_{epoch}.jpg")

        print(f"train_loss: {round(train_avg_loss, 4)}, valid_loss: {round(valid_avg_loss, 4)}, \
               pos_distance: {round(pos_distance_mean, 4)}, neg_distance: {round(neg_distance_mean, 4)}, \
               next_lr: {opt.param_groups[0]['lr']}")
        