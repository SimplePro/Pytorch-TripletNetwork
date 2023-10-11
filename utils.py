from torchvision.datasets import MNIST
from torch.utils.data import DataLoader 
from torch.utils.data import TensorDataset
import torchvision.transforms as TF

from random import randint, choice


class TripletDataset(TensorDataset):

    def __init__(self, dataset, class_n, transform):

        super().__init__()

        self.dataset = []
        self.class_n = class_n
        self.transform = transform

        self.class_indexes = [[] for _ in range(self.class_n)]

        for i, (x, y) in enumerate(dataset):
            self.class_indexes[y].append(i)
            self.dataset.append([x, y])


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        
        data = self.dataset[index]

        pos_index = choice(self.class_indexes[data[1]])
        neg_index = choice(self.class_indexes[data[1]-randint(1, self.class_n-1)])

        pos_data = self.dataset[pos_index]
        neg_data = self.dataset[neg_index]

        data[0] = self.transform(data[0])
        pos_data[0] = self.transform(pos_data[0])
        neg_data[0] = self.transform(neg_data[0])

        return *data, *pos_data, *neg_data
        
        

if __name__ == '__main__':
    mnist = MNIST(root="./mnist", download=True)
    triplet_dataset = TripletDataset(dataset=mnist, class_n=10, transform=TF.Compose([]))

    print(triplet_dataset[0])