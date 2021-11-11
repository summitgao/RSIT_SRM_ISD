
import argparse
import itertools
import os
import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from model import Generator
from model import StyleDiscriminator

from utils import ReplayBuffer
from utils import LambdaLR
from datasets import ImageDataset


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='use instance normalization')
parser.add_argument('--dataroot', type=str, default='datasets/yourdatasets/',
                    help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--save_dir', default='./output', help='directory to save the model')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if not os.path.exists(opt.save_dir):
    os.mkdir(opt.save_dir)

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = StyleDiscriminator(opt.input_nc)
netD_B = StyleDiscriminator(opt.output_nc)


if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()


# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
l1_loss = torch.nn.L1Loss()


optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))


lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)


# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()  # buffer res fake images
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
               transforms.RandomCrop(opt.size),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)


###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        #-----------------------------------
        #### Generators A2B and B2A ######
        #-----------------------------------
        netG_A2B.train()
        netG_B2A.train()

        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0
        loss_identity = loss_identity_A + loss_identity_B


        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake, _ = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)  # E[(D(G(x)) − 1)^2]

        fake_A = netG_B2A(real_B)
        pred_fake, _ = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)  # E[(D(G(x)) − 1)^2]
        loss_GAN = loss_GAN_A2B + loss_GAN_B2A


        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0
        loss_G_cycle = loss_cycle_ABA + loss_cycle_BAB

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        optimizer_G.step()


        #----------------------------#
        ###### Discriminator A #######
        #----------------------------#

        netD_A.train()
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real, _ = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)  # E[(D(x) − 1)^2]

        _, pred_real_S = netD_A(real_A)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)

        pred_fake, _ = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)  # E[D(G(y) - 0)^2]

        _, pred_fake_S = netD_A(fake_A.detach())
        loss_D_fake_S = l1_loss(pred_fake_S, pred_real_S)     #E[||D(G(y)) - D(x)||]

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)  # E[(D(x) − 1)^2] + E[D(G(y))^2]
        loss_D_SA = (loss_D_fake_S) * 10.0
        loss_D_A_all = loss_D_A + loss_D_SA
        loss_D_A_all.backward()
        optimizer_D_A.step()

        #----------------------------#
        ###### Discriminator B ######
        #----------------------------#

        netD_B.train()
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real, _ = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)  # E[(D(y) − 1)^2]

        _, pred_real_S = netD_B(real_B)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)

        pred_fake, _ = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)   # E[D(G(x) - 0)^2]

        _, pred_fake_S = netD_B(fake_B.detach())
        loss_D_fake_S = l1_loss(pred_fake_S, pred_real_S)  # E[||D(G(x)) - D(y)||]


        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)  # E[(D(y) − 1)^2] + E[D(G(x))^2]
        loss_D_SB = (loss_D_fake_S) * 10.0
        loss_D_B_all = loss_D_B + loss_D_SB
        loss_D_B_all.backward()
        optimizer_D_B.step()

        loss_D = loss_D_A_all + loss_D_B_all

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f][G loss: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),

            )
        )

    ## Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    ## Save models checkpoints
    torch.save(netG_A2B.state_dict(), '{:s}/netG_A2B.pth'.format(opt.save_dir))
    torch.save(netG_B2A.state_dict(), '{:s}/netG_B2A.pth'.format(opt.save_dir))
    # torch.save(netD_A.state_dict(), '{:s}/netD_A.pth'.format(opt.save_dir))
    # torch.save(netD_B.state_dict(), '{:s}/netD_B.pth'.format(opt.save_dir))

###################################
