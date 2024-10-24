import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

# class CustomDataset(Dataset):
#     def __init__(self, data_path, transform=None):
#         # Load dataset and apply any transforms (augmentations, normalization, etc.)
#         self.data = load_data(data_path)
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         if self.transform:
#             sample = self.transform(sample)
#         return sample

def get_dataloader(dataset_identifier,data_path, batch_size, shuffle=True):
    dataset = None
    if dataset_identifier == None:
        # then we pick the default example, FashionMNIST
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

        dataset = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
    else:
        # if none of the above worked
        raise Exception("invalid dataset identifier, maybe you spelled it wrong?")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)