import os
import torch
from torchvision import datasets, transforms


def load_train_dataset(image_size, dataroot, batch_size=1):
    dataset = datasets.ImageFolder(root=dataroot,
        transform=transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True)

    return dataloader