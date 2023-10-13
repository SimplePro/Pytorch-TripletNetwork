import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.utils import make_grid
import torchvision.transforms as TF
from torchvision.datasets import MNIST

from models import AutoEncoder

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import math

import sys
sys.path.append('/home/kdhsimplepro/kdhsimplepro/AI/TripletNet')

from utils import TripletDataset
from visualization_embedding_space import get_embedding, save_2d_embedding_space



DEVICE = "cuda"
EPOCHS = 30
LEARNING_RATE = 0.04
LEARNING_RATE_DECAY = 0.99
MOMENTUM = 0.9

@torch.no_grad()
def save_representation_space(model, dataset, class_n, save_path):
    class_embedding = get_embedding(dataset=dataset, class_n=class_n, model=model)
    save_2d_embedding_space(class_embedding, save_path)

@torch.no_grad()
@torch.inference_mode()
def save_pred(model, test_x, save_path):
    pred = model(test_x).cpu().detach()

    grid = make_grid(pred, nrow=int(math.sqrt(test_x.size(0))))
    TF.ToPILImage()(grid).save(save_path)

if __name__ == '__main__':

    model = AutoEncoder().to(DEVICE)
    
    transform = TF.Compose([
        TF.ToTensor(),
        # TF.Normalize([0], [1]) # this is a problem (makes some noisy reconstructed images and forces ReLU function not to be used)
    ])

    train_mnist = MNIST(root="../mnist", train=True, download=True, transform=transform)
    valid_mnist = MNIST(root="../mnist", train=False, download=True, transform=transform)

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

    for (x, _, _, _, _, _) in validloader:
        test_x = x.to(DEVICE)
        break

    opt = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=LEARNING_RATE_DECAY, verbose=False)

    train_history = []
    valid_history = []

    best_loss = 1e8
    best_state_dict = None

    for epoch in range(EPOCHS):
        print("-" * 50 + "EPOCH:", epoch, "-" * 50)
        train_avg_loss = 0
        valid_avg_loss = 0

        model.train()
        for (x, _, x_pos, _, x_neg, _) in trainloader:
            x = x.to(DEVICE)

            pred = model(x)

            loss = F.mse_loss(pred, x)

            train_history.append(loss.item())
            train_avg_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

        lr_scheduler.step()
        train_avg_loss /= len(trainloader)

        with torch.no_grad():
            for (x, _, x_pos, _, x_neg, _) in validloader:
                x = x.to(DEVICE)

                pred = model(x)

                loss = F.mse_loss(pred, x)

                valid_history.append(loss.item())
                valid_avg_loss += loss.item()

            valid_avg_loss /= len(validloader)

            if valid_avg_loss <= best_loss:
                best_loss = valid_avg_loss
                best_state_dict = model.state_dict()

                torch.save(best_state_dict, "./autoencoder_best_state_dict.pt")
                
        figure, ax1 = plt.subplots()

        plt.title("Loss History (blue: train, red: valid)")

        ax1.plot(range(len(train_history)), train_history, label="train_loss", color="blue")
        ax1.tick_params(axis='x', labelcolor='blue')
        
        ax2 = ax1.twiny()
        ax2.plot(range(len(valid_history)), valid_history, label="valid_loss", color="red")
        ax2.tick_params(axis="x", labelcolor="red")

        plt.savefig("./loss_history.jpg")
        plt.cla()

        save_representation_space(model=model, dataset=valid_mnist, class_n=10, save_path=f"../embedding_space/autoencoder/epoch_{epoch}.jpg")
        save_pred(model=model, test_x=test_x, save_path=f"./pred/epoch_{epoch}.jpg")

        print(f"train_loss: {round(train_avg_loss, 4)}, valid_loss: {round(valid_avg_loss, 4)}, \
               next_lr: {opt.param_groups[0]['lr']}")
        