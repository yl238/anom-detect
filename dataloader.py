import os
import torch
from torchvision import transforms, datasets

def load_vae_test_datasets(input_size, data):
    """
    load the datasets from vae_test folder
    """
    testdir = os.path.join(data, 'vae_test')

    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(testdir, transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                               shuffle=True, num_workers=4,
                                               pin_memory=True)
    return test_loader

def load_vae_train_datasets(input_size, data, batch_size):
    """
    load the datasets from vae_train folders
    """
    traindir = os.path.join(data, 'vae_train/train')
    valdir = os.path.join(data, 'vae_train/val')

    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = datasets.ImageFolder(traindir, transform)
    val_dataset = datasets.ImageFolder(valdir, transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=4,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=4,
                                             pin_memory=True)
    return train_loader, val_loader