import torch

import torchvision.transforms as TF
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np

from models import TripletNetwork


DEVICE = "cuda"

@torch.no_grad()
def get_embedding(dataset, class_n, model):
    class_embedding = [[] for _ in range(class_n)]

    for (x, y) in dataset:
        x = x.to(DEVICE)

        rep = model.get_rep(x.unsqueeze(0))
        class_embedding[y].append(rep.squeeze().cpu().numpy())

    return class_embedding


def save_2d_embedding_space(class_embedding, save_path):
    
    embeddings = []
    for emb in class_embedding:
        embeddings += emb

    embeddings = np.stack(embeddings)

    scaler = StandardScaler()
    scaler.fit(embeddings)

    pca = PCA(n_components=2)
    pca.fit(scaler.transform(embeddings))

    plt.figure(figsize=(18, 12))
    
    for cls_ in range(len(class_embedding)):
        embedding_2d = pca.transform(scaler.transform(class_embedding[cls_]))
        plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], label=str(cls_), s=2)
    
    plt.legend()
    plt.title("2d Feature Representation")
    plt.savefig(save_path)
    plt.cla()



if __name__ == '__main__':

    model = TripletNetwork(model="mnist").to(DEVICE)
    model.load_state_dict(torch.load("./triplet_best_state_dict.pt"))

    class_embedding = get_embedding(
        dataset=MNIST(
            root="./mnist",
            train=False,
            transform=TF.Compose([TF.ToTensor(), TF.Normalize([0], [1])])),
        class_n=10,
        model=model
    )

    save_2d_embedding_space(class_embedding, save_path="2d_feature_representation.jpg")