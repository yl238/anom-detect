import torch
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dataloader import load_train_dataset

data_dir = '/Users/sueliu/projects/anom-detect/data/tar_dir/vae_train'
loader = load_train_dataset(128, data_dir, batch_size=64)

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

real_batch = next(iter(loader))
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.title('Training images')
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],
padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.savefig('test.png', bbox_inches='tight')
