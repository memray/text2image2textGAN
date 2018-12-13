import numpy as np
import torch
import yaml
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import Text2ImageDataset
from models.gan_factory import gan_factory
from utils import Utils, Logger
from PIL import Image
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pdb
import pickle


is_cuda = torch.cuda.is_available()

class CycleTrainer(object):
    def __init__(self, type, dataset, split, lr, diter, vis_screen, save_path, l1_coef, l2_coef, pre_trained_gen_A, pre_trained_disc_A, batch_size, num_workers, epochs, pre_trained_gen_B = False, pre_trained_disc_B = False):
        with open('config.yaml', 'r') as f:
            config = yaml.load(f)

        forward_type = 'gan'
        # forward gan
        if is_cuda:
            self.generator_A = torch.nn.DataParallel(gan_factory.generator_factory(forward_type).cuda())
            self.discriminator_A = torch.nn.DataParallel(gan_factory.discriminator_factory(forward_type).cuda())
        else:
            self.generator_A = torch.nn.DataParallel(gan_factory.generator_factory(forward_type))
            self.discriminator_A = torch.nn.DataParallel(gan_factory.discriminator_factory(forward_type))

        # inverse gan
        # TODO: pass these as parameters from runtime in the future
        inverse_type = 'inverse_gan'
        if is_cuda:
            self.generator_B = torch.nn.DataParallel(gan_factory.generator_factory(inverse_type).cuda())
            self.discriminator_B = torch.nn.DataParallel(gan_factory.discriminator_factory(inverse_type).cuda())
        else:
            self.generator_B = torch.nn.DataParallel(gan_factory.generator_factory(inverse_type))
            self.discriminator_B = torch.nn.DataParallel(gan_factory.discriminator_factory(inverse_type))

        if dataset == 'birds':
            self.dataset = Text2ImageDataset(config['birds_dataset_path'], split=split)
        elif dataset == 'flowers':
            self.dataset = Text2ImageDataset(config['flowers_dataset_path'], split=split)
        else:
            print('Dataset not supported, please select either birds or flowers.')
            exit()

        self.load_pretrained_weights(pre_trained_gen_A, pre_trained_disc_A, pre_trained_gen_B, pre_trained_disc_B)

        self.noise_dim = 100
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.beta1 = 0.5
        self.num_epochs = epochs
        self.DITER = diter

        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers)

        self.optim_D_A = torch.optim.Adam(self.discriminator_A.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optim_G_A = torch.optim.Adam(self.generator_A.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.optim_D_B = torch.optim.Adam(self.discriminator_B.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optim_G_B = torch.optim.Adam(self.generator_B.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.logger = Logger(vis_screen)
        self.checkpoints_path = './checkpoints/'
        self.save_path = save_path
        self.type = type

    def load_pretrained_weights(self, pre_trained_gen_A, pre_trained_disc_A, pre_trained_gen_B = False, pre_trained_disc_B = False):
        """ load pre trained weights for G_A, G_B, D_A, D_B """
        if pre_trained_disc_A:
            self.discriminator_A.load_state_dict(torch.load(pre_trained_disc_A))
        else:
            self.discriminator_A.apply(Utils.weights_init)

        if pre_trained_gen_A:
            self.generator_A.load_state_dict(torch.load(pre_trained_gen_A))
        else:
            self.generator_A.apply(Utils.weights_init)

        if pre_trained_disc_B:
            self.discriminator_B.load_state_dict(torch.load(pre_trained_disc_B))
        else:
            self.discriminator_B.apply(Utils.weights_init)

        if pre_trained_gen_B:
            self.generator_B.load_state_dict(torch.load(pre_trained_gen_B))
        else:
            self.generator_B.apply(Utils.weights_init)


    def train(self, cls=False):
        self._train_cycle_gan(cls)


    def _train_cycle_gan(self, cls):
        criterion = nn.BCELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        lambda_A = 2
        lambda_B = 2
        iteration = 0
        gen_A_losses = []
        gen_B_losses = []
        disc_A_losses = []
        disc_B_losses = []
        cycle_A_losses = []
        cycle_B_losses = []

        for epoch in tqdm(range(self.num_epochs)):
            for sample in tqdm(self.data_loader):
                # pdb.set_trace()
                iteration += 1
                # sample.keys() = dict_keys(['right_images', 'wrong_images', 'inter_embed', 'right_embed', 'txt'])
                right_images = sample['right_images'] # 64x3x64x64
                right_embed = sample['right_embed'] # 64x1024
                wrong_images = sample['wrong_images'] # 64x3x64x64

                if is_cuda:
                    right_images = Variable(right_images.float()).cuda()
                    right_embed = Variable(right_embed.float()).cuda()
                    wrong_images = Variable(wrong_images.float()).cuda()
                else:
                    right_images = Variable(right_images.float())
                    right_embed = Variable(right_embed.float())
                    wrong_images = Variable(wrong_images.float())

                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))

                # ======== One sided label smoothing ==========
                # Helps preventing the discriminator from overpowering the
                # generator adding penalty when the discriminator is too confident
                # =============================================
                smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1))

                if is_cuda:
                    real_labels = Variable(real_labels).cuda()
                    smoothed_real_labels = Variable(smoothed_real_labels).cuda()
                    fake_labels = Variable(fake_labels).cuda()
                else:
                    real_labels = Variable(real_labels)
                    smoothed_real_labels = Variable(smoothed_real_labels)
                    fake_labels = Variable(fake_labels)

                #----------------------------------------------------------------------------
                # Train the discriminator A
                self.discriminator_A.zero_grad()
                outputs, activation_real = self.discriminator_A(right_images, right_embed)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs

                if cls:
                    outputs, _ = self.discriminator_A(wrong_images, right_embed)
                    wrong_loss = criterion(outputs, fake_labels)
                    wrong_score = outputs

                if is_cuda:
                    noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                else:
                    noise = Variable(torch.randn(right_images.size(0), 100))
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator_A(right_embed, noise) # G_A output
                outputs, _ = self.discriminator_A(fake_images, right_embed)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                d_loss_A = real_loss + fake_loss

                if cls:
                    d_loss_A = real_loss + 0.5 * wrong_loss + 0.5 * fake_loss

                disc_A_losses.append(d_loss_A.data[0])

                d_loss_A.backward()
                self.optim_D_A.step()
                # -----------------------------------------------------------------------------
                # Training discriminator B

                self.discriminator_B.zero_grad()
                outputs, activation_real = self.discriminator_B(right_images, right_embed)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs

                if cls:
                    outputs, _ = self.discriminator_B(wrong_images, right_embed)
                    wrong_loss = criterion(outputs, fake_labels)
                    wrong_score = outputs

                if is_cuda:
                    noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                else:
                    noise = Variable(torch.randn(right_images.size(0), 100))
                noise = noise.view(noise.size(0), 100, 1, 1)
                # fake_images = self.inv_generator(right_embed, noise)
                # outputs, _ = self.inv_discriminator(fake_images, right_embed)
                fake_embed = self.generator_B(right_images)
                outputs, _ = self.discriminator_B(right_images, fake_embed)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                d_loss_B = real_loss + fake_loss

                if cls:
                    d_loss_B = real_loss + 0.5 * wrong_loss + 0.5 * fake_loss

                disc_B_losses.append(d_loss_B.data[0])

                d_loss_B.backward()
                self.optim_D_B.step()

                #------------------------------------------------------------------------------

                # Train the generator A and B
                self.generator_A.zero_grad()
                self.generator_B.zero_grad()


                if is_cuda:
                    noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                else:
                    noise = Variable(torch.randn(right_images.size(0), 100))

                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator_A(right_embed, noise)
                outputs, activation_fake = self.discriminator_A(fake_images, right_embed)
                _, activation_real = self.discriminator_A(right_images, right_embed)

                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)

                loss_G_A = criterion(outputs, real_labels)

                if is_cuda:
                    noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                else:
                    noise = Variable(torch.randn(right_images.size(0), 100))

                noise = noise.view(noise.size(0), 100, 1, 1)
                # fake_images = self.inv_generator(right_embed, noise)
                fake_embed = self.generator_B(right_images)
                # outputs, activation_fake = self.inv_discriminator(right_images, right_embed)
                outputs, activation_fake = self.discriminator_B(right_images, fake_embed)

                loss_G_B = criterion(outputs, real_labels) + self.l1_coef * l1_loss(fake_embed, right_embed)

                # getting forward and backward cycle losses

                rec_A = self.generator_B(fake_images)
                loss_cycle_A = l1_loss(rec_A, right_embed) * lambda_A

                rec_B = self.generator_A(fake_embed, noise)
                loss_cycle_B = l1_loss(rec_B, right_images) * lambda_B
                
                loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B

                gen_A_losses.append(loss_G_A.data[0])
                gen_B_losses.append(loss_G_B.data[0])
                cycle_A_losses.append(loss_cycle_A.data[0])
                cycle_B_losses.append(loss_cycle_B.data[0])


                loss_G.backward()

                self.optim_G_A.step()
                self.optim_G_B.step()

            # if (epoch) % 10 == 0:
            if (epoch+1) % 50 == 0:
                Utils.save_checkpoint(self.discriminator_A, self.generator_A, self.checkpoints_path, self.save_path, epoch)
                Utils.save_checkpoint(self.discriminator_B, self.generator_B, self.checkpoints_path, self.save_path, epoch, inverse=True)

        with open('gen_A_loss.pkl', 'wb') as gen_a_f, open('disc_A_loss.pkl', 'wb') as disc_a_f, open('gen_B_loss.pkl', 'wb') as gen_b_f, open('disc_B_loss.pkl', 'wb') as disc_b_f, open('cycle_a_loss.pkl', 'wb') as cycle_a_f, open('cycle_b_loss.pkl', 'wb') as cycle_b_f:
            pickle.dump(gen_A_losses, gen_a_f)
            pickle.dump(disc_A_losses, disc_a_f)
            pickle.dump(gen_B_losses, gen_b_f)
            pickle.dump(disc_B_losses, disc_b_f)
            pickle.dump(cycle_A_losses, cycle_a_f)
            pickle.dump(cycle_B_losses, cycle_b_f)


        x = list(range(len(gen_A_losses)))
        plt.plot(x, gen_A_losses, 'g-', label='gen A loss')
        plt.plot(x, disc_A_losses, 'b-', label='disc A loss')
        plt.legend()
        plt.savefig('gen_A_vs_disc_A.png')
        plt.clf()

        plt.plot(x, gen_B_losses, 'g-', label='gen B loss')
        plt.plot(x, disc_B_losses, 'b-', label='disc B loss')
        plt.legend()
        plt.savefig('gen_B_vs_disc_B.png')
        plt.clf()

        plt.plot(x, cycle_A_losses, 'g-', label = 'cycle A loss')
        plt.legend()
        plt.savefig('cycle_A_loss.png')
        plt.clf()

        plt.plot(x, cycle_B_losses, 'b-', label = 'cycle B loss')
        plt.legend()
        plt.savefig('cycle_B_loss.png')
        plt.clf()


    def predict(self):
        for sample in self.data_loader:
            right_images = sample['right_images']
            right_embed = sample['right_embed']
            txt = sample['txt']

            if not os.path.exists('results/{0}'.format(self.save_path)):
                os.makedirs('results/{0}'.format(self.save_path))

            if is_cuda:
                right_images = Variable(right_images.float()).cuda()
                right_embed = Variable(right_embed.float()).cuda()
            else:
                right_images = Variable(right_images.float())
                right_embed = Variable(right_embed.float())

            # Train the generator
            if is_cuda:
                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
            else:
                noise = Variable(torch.randn(right_images.size(0), 100))
            noise = noise.view(noise.size(0), 100, 1, 1)
            fake_images = self.generator_A(right_embed, noise)

            self.logger.draw(right_images, fake_images)

            for image, t in zip(fake_images, txt):
                im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                im.save('results/{0}/{1}.jpg'.format(self.save_path, t.replace("/", "")[:100]))
                print(t)

        print('done prediction')







