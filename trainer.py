import io

import numpy as np
import torch
import yaml
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils
from txt2image_dataset import Text2ImageDataset, collate_fn
from models.gan_factory import gan_factory
from utils import Utils
from PIL import Image
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import pdb
import pickle
from build_vocab import Vocabulary 
from caption_gan_model import Image2TextDiscriminator, Image2TextGenerator
from torch.nn.utils.rnn import *

is_cuda = torch.cuda.is_available()

# def to_var(x, volatile=False):
#     if torch.cuda.is_available():
#         x = x.cuda()
#     if volatile:
#         with torch.no_grad():
#             return Variable(x)
#     else:
#         return Variable(x)

class Trainer(object):
    def __init__(self, type, dataset, split, lr, diter, vis_screen, save_path, l1_coef, l2_coef,
                 pre_trained_gen, pre_trained_disc,
                 batch_size, num_workers, epochs,
                 pre_trained_disc_B, pre_trained_gen_B,
                 pre_trained_caption_gen, pre_trained_caption_disc,
                 caption_embed_size, caption_hidden_size, caption_num_layers,
                 caption_gen_pretrain_num_epochs, caption_disc_pretrain_num_epochs,
                 caption_initial_noise=False
                 ):
        # with open('config.yaml', 'r') as f:
        #     config = yaml.load(f)
        config = utils.load_config()

        # forward gan
        if is_cuda:
            self.generator = torch.nn.DataParallel(gan_factory.generator_factory('gan').cuda())
            self.discriminator = torch.nn.DataParallel(gan_factory.discriminator_factory('gan').cuda())
            self.generator2 = torch.nn.DataParallel(gan_factory.generator_factory('stage2_gan').cuda())
            self.discriminator2 = torch.nn.DataParallel(gan_factory.discriminator_factory('stage2_gan').cuda())
        else:
            self.generator = torch.nn.DataParallel(gan_factory.generator_factory('gan'))
            self.discriminator = torch.nn.DataParallel(gan_factory.discriminator_factory('gan'))
            self.generator2 = torch.nn.DataParallel(gan_factory.generator_factory('stage2_gan'))
            self.discriminator2 = torch.nn.DataParallel(gan_factory.discriminator_factory('stage2_gan'))

        if pre_trained_disc:
            print('Loading pre_trained_disc A from: %s' % os.path.abspath(pre_trained_disc))
            self.discriminator.load_state_dict(torch.load(pre_trained_disc))
        else:
            self.discriminator.apply(Utils.weights_init)

        if pre_trained_gen:
            print('Loading pre_trained_gen A from: %s' % os.path.abspath(pre_trained_gen))
            self.generator.load_state_dict(torch.load(pre_trained_gen))
        else:
            self.generator.apply(Utils.weights_init)

        if pre_trained_disc_B:
            print('Loading pre_trained_disc B from: %s' % os.path.abspath(pre_trained_disc_B))
            self.discriminator2.load_state_dict(torch.load(pre_trained_disc_B))
        else:
            self.discriminator2.apply(Utils.weights_init)

        if pre_trained_gen_B:
            print('Loading pre_trained_gen B from: %s' % os.path.abspath(pre_trained_gen_B))
            self.generator2.load_state_dict(torch.load(pre_trained_gen_B))
        else:
            self.generator2.apply(Utils.weights_init)

        if dataset == 'birds':
            with open(config['birds_vocab_path'], 'rb') as f:
                self.vocab = pickle.load(f)
            self.dataset = Text2ImageDataset(config['birds_dataset_path'], dataset_type='birds', vocab=self.vocab, split=split)
        elif dataset == 'flowers':
            with open(config['flowers_vocab_path'], 'rb') as f:
                self.vocab = pickle.load(f)
            self.dataset = Text2ImageDataset(config['flowers_dataset_path'], dataset_type='flowers', vocab=self.vocab, split=split)
        else:
            print('Dataset not supported, please select either birds or flowers.')
            exit()

        self.noise_dim = 100
        self.batch_size = batch_size
        self.lr = lr
        self.beta1 = 0.5
        self.num_epochs = epochs
        self.DITER = diter
        self.num_workers = num_workers

        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers, collate_fn=collate_fn)

        self.optimD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.optimD2 = torch.optim.Adam(self.discriminator2.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimG2 = torch.optim.Adam(self.generator2.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.checkpoints_path = './checkpoints/'
        self.save_path = save_path
        self.type = type

        # settings for caption GAN
        self.embed_size = caption_embed_size
        self.hidden_size = caption_hidden_size
        self.num_layers = caption_num_layers
        self.caption_initial_noise = caption_initial_noise

        self.gen_pretrain_num_epochs = caption_gen_pretrain_num_epochs
        self.disc_pretrain_num_epochs = caption_disc_pretrain_num_epochs

        self.figure_path = './figures/'

        if is_cuda:
            self.caption_generator = Image2TextGenerator(
                self.embed_size,
                self.hidden_size,
                len(self.vocab),
                self.num_layers,
                initial_noise=self.caption_initial_noise).cuda()
            self.caption_discriminator = Image2TextDiscriminator(self.embed_size, self.hidden_size, len(self.vocab), self.num_layers).cuda()
        else:
            self.caption_generator = Image2TextGenerator(
                self.embed_size,
                self.hidden_size,
                len(self.vocab),
                self.num_layers,
                initial_noise=self.caption_initial_noise)
            self.caption_discriminator = Image2TextDiscriminator(self.embed_size, self.hidden_size, len(self.vocab), self.num_layers)

        if pre_trained_caption_gen and os.path.exists(pre_trained_caption_gen):
            print('loaded pretrained caption generator')
            self.caption_generator.load_state_dict(torch.load(pre_trained_caption_gen))

        if pre_trained_caption_disc and os.path.exists(pre_trained_caption_disc):
            print('loaded pretrained caption discriminator')
            self.caption_discriminator.load_state_dict(torch.load(pre_trained_caption_disc))
        
        self.optim_captionG = torch.optim.Adam(list(self.caption_generator.parameters()))
        self.optim_captionD = torch.optim.Adam(list(self.caption_discriminator.parameters()))


    def train(self, cls=False, interp=False):
        if self.type == 'gan':
            self._train_gan(cls, interp)
        elif self.type == 'stackgan':
            self._train_stack_gan(cls, interp)
        elif self.type == 'pretrain_txt2img':
            self._pretrain_caption2image()
        elif self.type == 'pretrain_img2txt':
            self._pretrain_image2caption()

    def _pretrain_caption2image(self, cls, interp):
        '''
        Pretrain a text-encoder and an image-generator
        :param cls:
        :param interp:
        :return:
        '''

        criterion = nn.BCELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        iteration = 0

        # cycle gan params
        lambda_a = 2
        lambda_b = 2
        mle_criterion = nn.CrossEntropyLoss()

        gen_losses = []
        disc_losses = []
        cycle_a_losses = []
        for epoch in tqdm(range(self.num_epochs)):
            for sample in tqdm(self.data_loader):
                iteration += 1
                right_captions = sample['captions']
                right_lengths = sample['lengths']
                right_images = sample['right_images']  # 64x3x64x64
                right_images128 = sample['right_images128']  # 64x3x128x128

                wrong_captions = sample['wrong_captions']
                wrong_lengths = sample['wrong_lengths']  # 64x3x64x64
                wrong_images = sample['wrong_images']  # 64x3x64x64
                wrong_images128 = sample['wrong_images128']  # 64x3x128x128

                if is_cuda:
                    right_images = Variable(right_images.float()).cuda()
                    wrong_images = Variable(wrong_images.float()).cuda()
                    right_images128 = Variable(right_images128.float()).cuda()
                    wrong_images128 = Variable(wrong_images128.float()).cuda()
                    right_captions = Variable(right_captions.long()).cuda()
                else:
                    right_images = Variable(right_images.float())
                    wrong_images = Variable(wrong_images.float())
                    right_images128 = Variable(right_images128.float())
                    wrong_images128 = Variable(wrong_images128.float())
                    right_captions = Variable(right_captions.long())

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

                # Train the discriminator
                self.discriminator.zero_grad()

                # ------------------- Training D stage 1 -------------------------------
                outputs, activation_real = self.discriminator(right_images, right_embed)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs

                if cls:
                    outputs, _ = self.discriminator(wrong_images, right_embed)
                    wrong_loss = criterion(outputs, fake_labels)
                    wrong_score = outputs

                if is_cuda:
                    noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                else:
                    noise = Variable(torch.randn(right_images.size(0), 100))
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, _ = self.discriminator(fake_images, right_embed)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                if cls:
                    d_loss = real_loss + 0.5 * wrong_loss + 0.5 * fake_loss
                else:
                    d_loss = real_loss + fake_loss

                d_loss.backward()
                self.optimD.step()

                # -------------------- Training G stage 1 -------------------------------
                self.generator.zero_grad()
                self.discriminator.zero_grad()
                if is_cuda:
                    noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                else:
                    noise = Variable(torch.randn(right_images.size(0), 100))

                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, activation_fake = self.discriminator(fake_images, right_embed)

                g_loss = criterion(outputs, real_labels)

                if (interp):
                    """ GAN INT loss"""
                    available_batch_size = int(right_embed.size(0))
                    first_part = right_embed[:int(available_batch_size / 2), :]
                    second_part = right_embed[int(available_batch_size / 2):, :]
                    interp_embed = (first_part + second_part) * 0.5

                    if is_cuda:
                        noise = Variable(torch.randn(int(available_batch_size / 2), 100)).cuda()
                    else:
                        noise = Variable(torch.randn(int(available_batch_size), 100))

                    noise = noise.view(noise.size(0), 100, 1, 1)

                    interp_real_labels = torch.ones(int(available_batch_size / 2))
                    if is_cuda:
                        interp_real_labels = Variable(interp_real_labels).cuda()
                    else:
                        interp_real_labels = Variable(interp_real_labels)

                    fake_images = self.generator(interp_embed, noise)
                    outputs, activation_fake = self.discriminator(fake_images, interp_embed)
                    g_int_loss = criterion(outputs, interp_real_labels)
                    g_loss = g_loss + 0.2 * g_int_loss

                g_loss.backward()
                self.optimG.step()

                # -------------------- Training D stage 2 -------------------------------
                self.discriminator2.zero_grad()
                outputs = self.discriminator2(right_images128, right_embed)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs

                if cls:
                    outputs = self.discriminator2(wrong_images128, right_embed)
                    wrong_loss = criterion(outputs, fake_labels)
                    wrong_score = outputs

                if is_cuda:
                    noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                else:
                    noise = Variable(torch.randn(right_images.size(0), 100))

                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images_v1 = self.generator(right_embed, noise)
                fake_images_v1 = fake_images_v1.detach()
                fake_images = self.generator2(fake_images_v1, right_embed)
                fake_images = fake_images.detach()
                outputs = self.discriminator2(fake_images, right_embed)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                if cls:
                    d_loss2 = real_loss + 0.5 * wrong_loss + 0.5 * fake_loss
                else:
                    d_loss2 = real_loss + fake_loss

                d_loss2.backward()
                self.optimD2.step()

                # -------------------- Training G stage 2 -------------------------------
                self.generator2.zero_grad()
                self.discriminator2.zero_grad()
                if is_cuda:
                    noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                else:
                    noise = Variable(torch.randn(right_images.size(0), 100))

                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images_v1 = self.generator(right_embed, noise)
                fake_images_v1 = fake_images_v1.detach()
                fake_images = self.generator2(fake_images_v1, right_embed)
                outputs = self.discriminator2(fake_images, right_embed)

                g_loss2 = criterion(outputs, real_labels)
                g_loss2.backward()
                self.optimG2.step()

                gen_losses.append(g_loss2.data[0])
                disc_losses.append(d_loss2.data[0])

                # Generate caption with caption GAN (inverse GAN)
                # fake_images.requires_grad = False # freeze the caption generator
                self.caption_generator.zero_grad()
                sampled_captions, _ = self.caption_generator.forward(fake_images, right_captions, right_lengths)
                targets = pack_padded_sequence(right_captions, right_lengths, batch_first=True)[0]
                loss_cycle_A = mle_criterion(sampled_captions, targets) * lambda_a
                loss_cycle_A.backward()
                self.optimG2.step()
                self.optim_captionG.step()
                cycle_a_losses.append(loss_cycle_A.data[0])

            with open('gen.pkl', 'wb') as f_gen, open('disc.pkl', 'wb') as f_disc:
                pickle.dump(gen_losses, f_gen)
                pickle.dump(disc_losses, f_disc)

            if (epoch + 1) % 10 == 0:
                # if (epoch+1) % 5 == 0:
                Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, self.save_path,
                                      epoch + 1)
                Utils.save_checkpoint(self.discriminator2, self.generator2, self.checkpoints_path, self.save_path,
                                      epoch + 1, inverse=False, stage=2)
                torch.save(self.caption_discriminator.state_dict(), os.path.join(self.checkpoints_path, self.save_path,
                                                                                 'cycle_caption_disc-%d.pkl' % (
                                                                                             epoch + 1)))
                torch.save(self.caption_generator.state_dict(), os.path.join(self.checkpoints_path, self.save_path,
                                                                             'cycle_caption_gen-%d.pkl' % (epoch + 1)))

        # Plot pretraining figures
        plt.plot(disc_losses, label='stage 1 disc losses')
        plt.savefig(self.figure_path + 'stage_1_disc_losses.png')
        plt.clf()

        plt.plot(gen_losses, label='stage_1_gen_loss')
        plt.savefig(self.figure_path + 'stage_1_gen_losses.png')
        plt.clf()

        plt.plot(disc_losses, label='cycle_a_losses')
        plt.savefig(self.figure_path + 'cycle_a_losses.png')
        plt.clf()


    def _pretrain_image2caption(self):
        # Create model directory
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)
        
        if not os.path.exists(self.figure_path):
            os.makedirs(self.figure_path)

        # Build the models (Gen)
        # generator = CaptionGenerator(self.embed_size, self.hidden_size, len(self.vocab), self.num_layers)
        generator = self.caption_generator

        # Build the models (Disc)
        # discriminator = CaptionDiscriminator(self.embed_size, self.hidden_size, len(self.vocab), self.num_layers)
        discriminator = self.caption_discriminator

        if torch.cuda.is_available():
            generator.cuda()
            discriminator.cuda()

        # Loss and Optimizer (Gen)
        mle_criterion = nn.CrossEntropyLoss()
        params_gen = list(generator.parameters())
        optimizer_gen = torch.optim.Adam(params_gen)

        # Loss and Optimizer (Disc)
        params_disc = list(discriminator.parameters())
        optimizer_disc = torch.optim.Adam(params_disc)

        disc_losses = []
        gen_losses = []
        for epoch in tqdm(range(max([int(self.gen_pretrain_num_epochs), int(self.disc_pretrain_num_epochs)]))):
            for sample in tqdm(self.data_loader):
                images = sample['right_images128'] # 64x3x128x128
                captions = sample['captions'] # batch_size * max_length
                lengths = sample['lengths'] # batch_size * 1
                wrong_captions = sample['wrong_captions'] # batch_size * max_length
                wrong_lengths = sample['wrong_lengths'] # batch_size * 1

                if is_cuda:
                    images = Variable(images).float().cuda()
                    captions = Variable(captions).long().cuda()
                    wrong_captions = Variable(wrong_captions).long().cuda()
                else:
                    images = Variable(images).float()
                    captions = Variable(captions).long()
                    wrong_captions = Variable(wrong_captions).long()


                # Train generator with teacher forcing
                if epoch < int(self.gen_pretrain_num_epochs):
                    generator.zero_grad()
                    # get logits [batch_size, max_len, vocab_size] from decoder
                    #    and truncate the last token (output after EOS is useless)
                    logits, _ = generator(images, captions, lengths)
                    logits = logits[:,1:,:].contiguous().view(-1, logits.shape[-1])
                    # ignore the 1st token (BOS) in target captions, [batch_size, max_len-1]
                    targets = captions[:,1:].contiguous().view(-1)
                    loss_gen = mle_criterion(logits, targets)
                    gen_losses.append(float(loss_gen.cpu().data.numpy()))
                    loss_gen.backward()
                    optimizer_gen.step()

                # Train discriminator with a positive caption and a random negative caption
                if epoch < int(self.disc_pretrain_num_epochs):
                    discriminator.zero_grad()
                    rewards_real = discriminator(images, captions, lengths)
                    # rewards_fake = discriminator(images, sampled_captions, sampled_lengths) 
                    rewards_wrong = discriminator(images, wrong_captions, wrong_lengths)
                    real_loss = -torch.mean(torch.log(rewards_real))
                    # fake_loss = -torch.mean(torch.clamp(torch.log(1 - rewards_fake), min=-1000))
                    wrong_loss = -torch.mean(torch.clamp(torch.log(1 - rewards_wrong), min=-1000))
                    loss_disc = real_loss + wrong_loss # + fake_loss, no fake_loss because this is pretraining

                    disc_losses.append(float(loss_disc.cpu().data.numpy()))
                    loss_disc.backward()
                    optimizer_disc.step()

            if (epoch + 1) % 10 == 0:
                Utils.save_checkpoint(discriminator, generator, self.checkpoints_path,
                                      self.save_path, epoch + 1)

        # Save pretrained models
        torch.save(discriminator.state_dict(),
                   os.path.join(self.checkpoints_path,
                                'pretrained-img2txt-discriminator-%d.pkl' % int(self.disc_pretrain_num_epochs)))
        torch.save(generator.state_dict(),
                   os.path.join(self.checkpoints_path,
                                'pretrained-img2txt-generator-%d.pkl' % int(self.gen_pretrain_num_epochs)))

        # Plot pretraining figures
        plt.plot(disc_losses, label='pretraining_caption_disc_loss')
        plt.savefig(self.figure_path + 'pretraining_img2txt_disc_losses.png')
        plt.clf()

        plt.plot(gen_losses, label='pretraining_gen_loss')
        plt.savefig(self.figure_path + 'pretraining_img2txt_gen_losses.png')
        plt.clf()


    def _train_gan(self, cls, interp):
        criterion = nn.BCELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        iteration = 0

        gen_losses = []
        disc_losses = []
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

                # Train the discriminator
                self.discriminator.zero_grad()
                outputs, activation_real = self.discriminator(right_images, right_embed)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs

                if cls:
                    outputs, _ = self.discriminator(wrong_images, right_embed)
                    wrong_loss = criterion(outputs, fake_labels)
                    wrong_score = outputs

                if is_cuda:
                    noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                else:
                    noise = Variable(torch.randn(right_images.size(0), 100))
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, _ = self.discriminator(fake_images, right_embed)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                if cls:
                    d_loss = real_loss + 0.5 * wrong_loss + 0.5 * fake_loss
                else:
                    d_loss = real_loss + fake_loss

                d_loss.backward()
                self.optimD.step()

                # Train the generator
                self.generator.zero_grad()
                if is_cuda:
                    noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                else:
                    noise = Variable(torch.randn(right_images.size(0), 100))

                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, activation_fake = self.discriminator(fake_images, right_embed)
                _, activation_real = self.discriminator(right_images, right_embed)

                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)



                #======= Generator Loss function============
                # This is a customized loss function, the first term is the regular cross entropy loss
                # The second term is feature matching loss, this measure the distance between the real and generated
                # images statistics by comparing intermediate layers activations
                # The third term is L1 distance between the generated and real images, this is helpful for the conditional case
                # because it links the embedding feature vector directly to certain pixel values.
                #===========================================
                g_loss = criterion(outputs, real_labels) \
                + self.l2_coef * l2_loss(activation_fake, activation_real.detach()) \
                + self.l1_coef * l1_loss(fake_images, right_images)

                if (interp):
                    """ GAN INT loss"""
                    # pdb.set_trace()
                    # print('iter {}, size {}, right {}'.format(iteration, self.batch_size, right_embed.size()))i
                    available_batch_size = int(right_embed.size(0))
                    first_part = right_embed[:int(available_batch_size/2),:]
                    second_part = right_embed[int(available_batch_size/2):,:]
                    interp_embed = (first_part + second_part)*0.5
                    
                    if is_cuda:
                        noise = Variable(torch.randn(int(available_batch_size/2), 100)).cuda()
                    else:
                        noise = Variable(torch.randn(int(available_batch_size), 100))

                    interp_real_labels = torch.ones(int(available_batch_size/2))
                    if is_cuda:
                        interp_real_labels = Variable(interp_real_labels).cuda()
                    else:
                        interp_real_labels = Variable(interp_real_labels)

                    fake_images = self.generator(interp_embed, noise)
                    outputs, activation_fake = self.discriminator(fake_images, interp_embed)
                    g_int_loss = criterion(outputs, interp_real_labels)
                    g_loss = g_loss + 0.2 * g_int_loss

                g_loss.backward()
                self.optimG.step()

                gen_losses.append(g_loss.data[0])
                disc_losses.append(d_loss.data[0])

            with open('gen.pkl', 'wb') as f_gen, open('disc.pkl', 'wb') as f_disc:
                pickle.dump(gen_losses, f_gen)
                pickle.dump(disc_losses, f_disc)


            x = list(range(len(gen_losses)))
            plt.plot(x, gen_losses, 'g-', label='gen loss')
            plt.plot(x, disc_losses, 'b-', label='disc loss')
            plt.legend()
            plt.savefig('gen_vs_disc_.png')
            plt.clf()

            if (epoch + 1) % 10 == 0:
                Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, self.save_path, epoch)


    def _train_stack_gan(self, cls, interp):

        criterion = nn.BCELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        iteration = 0

        # cycle gan params
        lambda_a = 2
        lambda_b = 2
        mle_criterion = nn.CrossEntropyLoss() 

        gen_losses = []
        disc_losses = []
        cycle_a_losses = []
        for epoch in tqdm(range(self.num_epochs)):
            for sample in tqdm(self.data_loader):
                # pdb.set_trace()
                iteration += 1
                # sample.keys() = dict_keys(['right_images', 'wrong_images', 'inter_embed', 'right_embed', 'txt'])
                right_images = sample['right_images'] # 64x3x64x64
                right_embed = sample['right_embed'] # 64x1024
                wrong_images = sample['wrong_images'] # 64x3x64x64
                right_images128 = sample['right_images128'] # 64x3x128x128
                wrong_images128 = sample['wrong_images128'] # 64x3x128x128
                right_captions = sample['captions']
                right_lengths = sample['lengths']

                if is_cuda:
                    right_images = Variable(right_images.float()).cuda()
                    right_embed = Variable(right_embed.float()).cuda()
                    wrong_images = Variable(wrong_images.float()).cuda()
                    right_images128 = Variable(right_images128.float()).cuda()
                    wrong_images128 = Variable(wrong_images128.float()).cuda()
                    right_captions = Variable(right_captions.long()).cuda()
                else:
                    right_images = Variable(right_images.float())
                    right_embed = Variable(right_embed.float())
                    wrong_images = Variable(wrong_images.float())
                    right_images128 = Variable(right_images128.float())
                    wrong_images128 = Variable(wrong_images128.float())
                    right_captions = Variable(right_captions.long())

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

                # Train the discriminator
                self.discriminator.zero_grad()

                # ------------------- Training D stage 1 -------------------------------
                outputs, activation_real = self.discriminator(right_images, right_embed)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs

                if cls:
                    outputs, _ = self.discriminator(wrong_images, right_embed)
                    wrong_loss = criterion(outputs, fake_labels)
                    wrong_score = outputs

                if is_cuda:
                    noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                else:
                    noise = Variable(torch.randn(right_images.size(0), 100))
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, _ = self.discriminator(fake_images, right_embed)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                if cls:
                    d_loss = real_loss + 0.5 * wrong_loss + 0.5 * fake_loss
                else:
                    d_loss = real_loss + fake_loss

                d_loss.backward()
                self.optimD.step()

                # -------------------- Training G stage 1 -------------------------------
                self.generator.zero_grad()
                self.discriminator.zero_grad()
                if is_cuda:
                    noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                else:
                    noise = Variable(torch.randn(right_images.size(0), 100))

                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, activation_fake = self.discriminator(fake_images, right_embed)

                g_loss = criterion(outputs, real_labels)

                if (interp):
                    """ GAN INT loss"""
                    available_batch_size = int(right_embed.size(0))
                    first_part = right_embed[:int(available_batch_size/2),:]
                    second_part = right_embed[int(available_batch_size/2):,:]
                    interp_embed = (first_part + second_part)*0.5
                    
                    if is_cuda:
                        noise = Variable(torch.randn(int(available_batch_size/2), 100)).cuda()
                    else:
                        noise = Variable(torch.randn(int(available_batch_size), 100))

                    noise = noise.view(noise.size(0), 100, 1, 1)

                    interp_real_labels = torch.ones(int(available_batch_size/2))
                    if is_cuda:
                        interp_real_labels = Variable(interp_real_labels).cuda()
                    else:
                        interp_real_labels = Variable(interp_real_labels)

                    fake_images = self.generator(interp_embed, noise)
                    outputs, activation_fake = self.discriminator(fake_images, interp_embed)
                    g_int_loss = criterion(outputs, interp_real_labels)
                    g_loss = g_loss + 0.2 * g_int_loss

                g_loss.backward()
                self.optimG.step()

                # -------------------- Training D stage 2 -------------------------------
                self.discriminator2.zero_grad()
                outputs = self.discriminator2(right_images128, right_embed)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs

                if cls:
                    outputs = self.discriminator2(wrong_images128, right_embed)
                    wrong_loss = criterion(outputs, fake_labels)
                    wrong_score = outputs

                if is_cuda:
                    noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                else:
                    noise = Variable(torch.randn(right_images.size(0), 100))

                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images_v1 = self.generator(right_embed, noise)
                fake_images_v1 = fake_images_v1.detach()
                fake_images = self.generator2(fake_images_v1, right_embed)
                fake_images = fake_images.detach()
                outputs = self.discriminator2(fake_images, right_embed)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                if cls:
                    d_loss2 = real_loss + 0.5 * wrong_loss + 0.5 * fake_loss
                else:
                    d_loss2 = real_loss + fake_loss

                d_loss2.backward()
                self.optimD2.step()


                # -------------------- Training G stage 2 -------------------------------
                self.generator2.zero_grad()
                self.discriminator2.zero_grad()
                if is_cuda:
                    noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                else:
                    noise = Variable(torch.randn(right_images.size(0), 100))

                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images_v1 = self.generator(right_embed, noise)
                fake_images_v1 = fake_images_v1.detach()
                fake_images = self.generator2(fake_images_v1, right_embed)
                outputs = self.discriminator2(fake_images, right_embed)

                g_loss2 = criterion(outputs, real_labels)
                g_loss2.backward()
                self.optimG2.step()

                gen_losses.append(g_loss2.data[0])
                disc_losses.append(d_loss2.data[0])

                # Generate caption with caption GAN (inverse GAN)
                # fake_images.requires_grad = False # freeze the caption generator
                self.caption_generator.zero_grad()
                sampled_captions, _ = self.caption_generator.forward(fake_images, right_captions, right_lengths)
                targets = pack_padded_sequence(right_captions, right_lengths, batch_first=True)[0]
                loss_cycle_A = mle_criterion(sampled_captions, targets)* lambda_a
                loss_cycle_A.backward()
                self.optimG2.step()
                self.optim_captionG.step()
                cycle_a_losses.append(loss_cycle_A.data[0])

            with open('gen.pkl', 'wb') as f_gen, open('disc.pkl', 'wb') as f_disc:
                pickle.dump(gen_losses, f_gen)
                pickle.dump(disc_losses, f_disc)

            if (epoch + 1) % 10 == 0:
            # if (epoch+1) % 5 == 0:
                Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, self.save_path, epoch + 1)
                Utils.save_checkpoint(self.discriminator2, self.generator2, self.checkpoints_path, self.save_path, epoch + 1, inverse=False, stage=2)
                torch.save(self.caption_discriminator.state_dict(), os.path.join(self.checkpoints_path, self.save_path, 'cycle_caption_disc-%d.pkl' % (epoch + 1)))
                torch.save(self.caption_generator.state_dict(), os.path.join(self.checkpoints_path, self.save_path, 'cycle_caption_gen-%d.pkl' % (epoch + 1)))

        # Plot pretraining figures
        plt.plot(disc_losses, label='stage 1 disc losses')
        plt.savefig(self.figure_path + 'stage_1_disc_losses.png')
        plt.clf()

        plt.plot(gen_losses, label='stage_1_gen_loss')
        plt.savefig(self.figure_path + 'stage_1_gen_losses.png')
        plt.clf()

        plt.plot(disc_losses, label='cycle_a_losses')
        plt.savefig(self.figure_path + 'cycle_a_losses.png')
        plt.clf()


    def predict(self, gan_type='gan'):
        torch.manual_seed(7)
        count = 0
        for sample in tqdm(self.data_loader):
            right_images = sample['right_images']
            right_images128 = sample['right_images128']
            right_images_original = sample['right_images_original']
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
            fake_images_A = self.generator(right_embed, noise)

            if (gan_type=='stackgan'):
                fake_images_B = self.generator2(fake_images_A, right_embed)

            for example_id in range(len(txt)):
                right_image = right_images128[example_id]
                right_image_original = right_images_original[example_id]
                image_A = fake_images_A[example_id]
                t = txt[example_id].strip().replace("/", "")[:200]

                im = Image.fromarray(right_image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                im.save('results/{0}/{1}_GT128.jpg'.format(self.save_path, t))

                im = right_image_original
                im.save('results/{0}/{1}_GT.jpg'.format(self.save_path, t))

                im = Image.fromarray(image_A.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                im.save('results/{0}/{1}_A.jpg'.format(self.save_path, t))

                if (gan_type == 'stackgan'):
                    image_B = fake_images_B[example_id]
                    im = Image.fromarray(image_B.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                    im.save('results/{0}/{1}_B.jpg'.format(self.save_path, t))

                print(t)

            count += 1
            # if count == 1:
            #     break






