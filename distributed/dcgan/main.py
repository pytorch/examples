"""
Distributed DCGAN example.
"""

from __future__ import print_function

import sys
import argparse
import os
import random

import torch
from torch import nn, optim
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets as dset, transforms
import torchvision.utils as vutils


def get_dataset(opt):
    """Create and return the appropriate dataset based on configuration.

    Args:
        opt: argparse.Namespace with dataset configuration.
    """
    if opt.dataroot is None and str(opt.dataset).lower() != 'fake':
        raise ValueError(f"`dataroot` parameter is required for dataset \"{opt.dataset}\"")

    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # Folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(opt.imageSize),
                                       transforms.CenterCrop(opt.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif opt.dataset == 'lsun':
        classes = [c + '_train' for c in opt.classes.split(',')]
        dataset = dset.LSUN(root=opt.dataroot, classes=classes,
                            transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(opt.imageSize),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,)),
                             ]))
    elif opt.dataset == 'fake':
        dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                                transform=transforms.ToTensor())
    else:
        raise ValueError(f"Unknown dataset: {opt.dataset}")

    return dataset


class Generator(nn.Module):
    """DCGAN generator network."""

    def __init__(self, nz, ngf, nc):
        super().__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        output = self.main(x)
        return output


class Discriminator(nn.Module):
    """DCGAN discriminator network."""

    def __init__(self, ndf, nc):
        super().__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)
        return output.view(-1, 1).squeeze(1)


def weights_init(m):
    """Custom weights initialization called on netG and netD."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """Verify that there are at least `min_gpus` available."""
    has_gpu = torch.accelerator.is_available()
    gpu_count = torch.accelerator.device_count()
    return has_gpu and gpu_count >= min_gpus


def setup_ddp(rank):
    """Setup distributed data parallel configuration."""
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    torch.accelerator.set_device_index(rank)

    # Initialize the process group
    dist.init_process_group(backend=backend)


def run_training(rank, world_size, opt):
    """Run training process for a specific rank."""
    # Setup DDP
    setup_ddp(rank)

    device = torch.accelerator.current_accelerator()
    getattr(torch, device.type).manual_seed_all(opt.manualSeed)
        
    # Create dataset and dataloader
    dataset = get_dataset(opt)
    
    # Use DistributedSampler for DDP
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=opt.batchSize, 
        sampler=sampler,
        num_workers=int(opt.workers), 
        pin_memory=True
    )

    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = 3

    # Create the networks
    netG = Generator(nz,ngf,nc).to(rank)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    netD = Discriminator(ndf, nc).to(rank)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    # Wrap with DDP
    netG = DDP(netG, device_ids=[rank])
    netD = DDP(netD, device_ids=[rank])
    
    # Loss function
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    if opt.dry_run:
        opt.niter = 1

    # Training loop
    for epoch in range(opt.niter):
        # Set epoch for DistributedSampler
        sampler.set_epoch(epoch)

        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with real
            netD.zero_grad()
            real_cpu = data[0].to(rank)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, dtype=real_cpu.dtype, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with fake
            noise = torch.randn(batch_size, opt.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            msg = (
                f"[{epoch}/{opt.niter}][{i}/{len(dataloader)}][Rank {rank}] "
                f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
            )
            print(msg)

            # Save samples periodically
            if i % 100 == 0:
                vutils.save_image(
                    real_cpu, f"{opt.outf}/real_samples.png", normalize=True
                )
                fake = netG(fixed_noise)
                vutils.save_image(
                    fake.detach(), f"{opt.outf}/fake_samples_epoch_{epoch:03d}.png",
                    normalize=True,
                )
            
            if opt.dry_run:
                break

        # Save checkpoints
        torch.save(netG.module.state_dict(), f"{opt.outf}/netG_epoch_{epoch}.pth")
        torch.save(netD.module.state_dict(), f"{opt.outf}/netD_epoch_{epoch}.pth")
    
    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    MIN_GPU_COUNT = 2
    if not verify_min_gpu_count(min_gpus=MIN_GPU_COUNT):
        print(
            f"Unable to locate sufficient {MIN_GPU_COUNT} gpus to run this example. Exiting."
        )

        sys.exit()

    parser = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=35)
    )
    parser.add_argument(
        '--dataset', required=True,
        help=(
            'cifar10 | lsun | mnist |imagenet | folder | lfw | fake'
        ),
    )
    parser.add_argument('--dataroot', required=False, help='path to dataset')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=2
    )
    parser.add_argument(
        '--batchSize', type=int, default=64, help='input batch size'
    )
    parser.add_argument(
        '--imageSize', type=int, default=64,
        help='the height / width of the input image to network',
    )
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help='number of generator filters')
    parser.add_argument(
        '--ndf', type=int, default=64,
        help='number of discriminator filters',
    )
    parser.add_argument(
        '--niter', type=int, default=25, help='number of epochs to train for'
    )
    parser.add_argument(
        '--lr', type=float, default=0.0002, help='learning rate, default=0.0002'
    )
    parser.add_argument(
        '--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5'
    )
    parser.add_argument(
        '--dry-run', action='store_true', help='check a single training cycle works'
    )
    parser.add_argument('--netG', default='', help='path to netG (to continue training)')
    parser.add_argument('--netD', default='', help='path to netD (to continue training)')
    parser.add_argument(
        '--outf', default='.',
        help='folder to output images and model checkpoints',
    )
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument(
        '--classes', default='bedroom',
        help='comma separated list of classes for the lsun data set',
    )
    
    opt = parser.parse_args()

    # Create output directory
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    # Set random seeds for reproducibility
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    rank = int(env_dict['RANK'])
    world_size = int(env_dict['WORLD_SIZE'])
    
    # Start distributed training
    run_training(rank, world_size, opt)
