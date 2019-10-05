from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from models.models import Generator , Discriminator
from IPython.display import HTML

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def main():

    parser = argparse.ArgumentParser(description='Train Cartoon avatar Gan models')
    parser.add_argument('--crop_size', default=64, type=int, help='Training images crop size')
    parser.add_argument('--num_epochs', default=50, type=int, help='Train epoch number')
    parser.add_argument('--data_root', default='data/cartoon', help='Root directory for dataset')
    parser.add_argument('--worker', default=2, type = int , help='Number of workers for dataloader')
    parser.add_argument('--batch_size', default=16, type = int , help='Batch size during training')
    parser.add_argument('--channels', default=3, type = int , help='Number of channels in the training images')
    parser.add_argument('--nz', default=100, type = int , help='Size of generator input')
    parser.add_argument('--ngf', default=64, type = int , help='Size of feature maps in generator')
    parser.add_argument('--ndf', default=64, type = int , help='Size of feature maps in descriminator')
    parser.add_argument('--lr', default=0.0002, type = float , help='Learning rate for optimizer')
    parser.add_argument('--beta1', default=0.5, type = float , help='Beta1 hyperparam for Adam optimizers')
    parser.add_argument('--beta2', default=0.999, type = float , help='Beta2 hyperparam for Adam optimizers')
    parser.add_argument('--ngpu', default=1, type = int , help='Number of GPUs , use 0 for CPU mode')
    parser.add_argument('--latent_vector_num', default=8, type = int , help='latent vectors that we will use to visualize , 8 means that it will visualize 8 images during training')
    opt = parser.parse_args()

    dataroot = opt.data_root
    workers = opt.worker
    batch_size = opt.batch_size
    image_size = opt.crop_size
    nc = opt.channels
    nz = opt.nz
    ngf = opt.ngf
    ndf = opt.ndf
    num_epochs = opt.num_epochs
    lr = opt.lr
    beta1 = opt.beta1
    beta2 = opt.beta2
    ngpu = opt.ngpu
    latent_vector_num = opt.latent_vector_num

    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)


    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Create the generator
    netG = Generator(ngpu, nz, ngf, nc).to(device)
    # Create the Discriminator
    netD = Discriminator(ngpu, nc, ndf).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))


    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)    
    netD.apply(weights_init)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Print models
    print(netG)
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    fixed_noise = torch.randn(latent_vector_num, nz, 1, 1, device=device)

    #real and fake labels during training
    real_label = 1
    fake_label = 0

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training ...")

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  
            # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                    # Save model data
                torch.save(netG.state_dict(), 'netG_epoch_%d.pth' % (iters))
                torch.save(netD.state_dict(), 'netD_epoch_%d.pth' % (iters))
                    # Print training stats
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 650 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1


    # Display and Save samples GIF
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save('output/samples.gif', writer='imagemagick', fps=100)


if __name__ == '__main__':
    main()








