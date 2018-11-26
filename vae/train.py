import os
import torch
import argparse
from loss import VAELoss
from torchvision.utils import make_grid
from utilities import trainVAE, validateVAE
from model import VAE
from dataloader import load_vae_train_datasets
from tensorboardX import SummaryWriter
import numpy as np

parser = argparse.ArgumentParser(description='VAE for outlier in skin image')
parser.add_argument('--data', metavar='DIR', help='path to dataset', type=str)

# for models
parser.add_argument('--image_size', default=256, type=int,
                    help='transformed image size, has to be power of 2')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')

# for optimization
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr_decay', default=0.1, type=float,
                    help='learning rate decay')
parser.add_argument('--schedule', type=int, nargs='+', default=[50],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--kl_weight', type=float, default=1,
                    help="weight on KL term")

# for checkpoint loading
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--reset_opt', action='store_true',
                    help='if true then we do not load optimizer')

# for logging
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--out_dir', default='./result', type=str,
                    help='output result for tensorboard and model checkpoint')
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

# load checkpoint
if args.resume is not None:
    checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
    print("checkpoint loaded!")
    print("val loss: {}\tepoch: {}\t".format(checkpoint['val_loss'], checkpoint['epoch']))

# model
model = VAE(args.image_size)
if args.resume is not None:
    model.load_state_dict(checkpoint['state_dict'])

# criterion
criterion = VAELoss(size_average=True, kl_weight=args.kl_weight)
if args.cuda is True:
    model = model.cuda()
    criterion = criterion.cuda()

# load data
train_loader, val_loader = load_vae_train_datasets(input_size=args.image_size,
                                                   data=args.data,
                                                   batch_size=args.batch_size)

# load optimizer and scheduler
opt = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
if args.resume is not None and not args.reset_opt:
    opt.load_state_dict(checkpoint['optimizer'])

scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.schedule,
                                                 gamma=args.lr_decay)

# make output dir
if os.path.isdir(args.out_dir):
    print("{} already exists!".format(args.out_dir))
os.mkdir(args.out_dir)

# save args
args_dict = vars(args)
with open(os.path.join(args.out_dir, 'config.txt'), 'w') as f:
    for k in args_dict.keys():
        f.write("{}:{}\n".format(k, args_dict[k]))
writer = SummaryWriter(log_dir=os.path.join(args.out_dir, 'logs'))

# main loop
best_loss = np.inf
for epoch in range(args.epochs):
    # train for one epoch
    scheduler.step()
    train_loss, train_kl, train_reconst_logp = trainVAE(train_loader, model, criterion, opt, epoch, args)
    writer.add_scalar('train_elbo', -train_loss, global_step=epoch + 1)
    writer.add_scalar('train_kl', train_kl, global_step=epoch + 1)
    writer.add_scalar('train_reconst_logp', train_reconst_logp, global_step=epoch + 1)

    # evaluate on validation set
    with torch.no_grad():
        val_loss, val_kl, val_reconst_logp = validateVAE(val_loader, model, criterion, args)
        writer.add_scalar('val_elbo', -val_loss, global_step=epoch + 1)
        writer.add_scalar('val_kl', val_kl, global_step=epoch + 1)
        writer.add_scalar('val_reconst_logp', val_reconst_logp, global_step=epoch + 1)

    # remember best acc and save checkpoint
    if val_loss < best_loss:
        print('checkpointed!')
        best_loss = val_loss
        save_dict = {'epoch': epoch + 1,
                     'state_dict': model.state_dict(),
                     'val_loss': val_loss,
                     'optimizer': opt.state_dict()}
        save_path = os.path.join(args.out_dir, 'best_model.pth.tar')
        torch.save(save_dict, save_path)
    print('curr lowest val loss {}'.format(best_loss))

    # visualize reconst and free sample
    print("plotting imgs...")
    with torch.no_grad():
        val_iter = val_loader.__iter__()

        # reconstruct 25 imgs
        imgs = val_iter._get_batch()[1][0][:25]
        if args.cuda:
            imgs = imgs.cuda()
        imgs_reconst, mu, logvar = model(imgs)

        # sample 25 imgs
        noises = torch.randn(25, model.nz, 1, 1)
        if args.cuda:
            noises = noises.cuda()
        samples = model.decode(noises)

        def write_image(tag, images):
            """
            write the resulting imgs to tensorboard.
            :param tag: The tag for tensorboard
            :param images: the torch tensor with range (-1, 1). [9, 3, 256, 256]
            """
            # make it from 0 to 255
            images = (images + 1) / 2
            grid = make_grid(images, nrow=5, padding=20)
            writer.add_image(tag, grid.detach(), global_step=epoch + 1)

        write_image("origin", imgs)
        write_image("reconst", imgs_reconst)
        write_image("samples", samples)
        print('done')

import ipdb
