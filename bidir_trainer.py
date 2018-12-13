import io

import numpy as np
import time
import torch
import yaml
import itertools
from torch import nn
from torch.autograd import Variable
# from utils.data import DataLoader
from torch.utils.data import DataLoader

from dataset import Text2ImageDataset, collate_fn
from model.gan_factory import gan_factory
import utils.utils as utils
from PIL import Image
import os
from tqdm import tqdm
tqdm.monitor_interval = 0
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import pdb
import pickle
from build_vocab import Vocabulary 
from model.crossmodal_gan_model import \
    TextDecoder, \
    ImageEncoder, TextEncoder, \
    ImageDecoder64, ImageDecoder128,\
    TextDiscriminator, \
    ImageDiscriminator, \
    ImageTextPairDiscriminator
from torch.nn.utils.rnn import *

is_cuda = torch.cuda.is_available()

class BiDirectionalTrainer(object):
    def __init__(self, type, config, dataset, split, lr, save_path, l1_coef, l2_coef,
                 batch_size, num_workers, epochs,
                 text_embed_dim, hidden_dim,
                 checkpoint, resume,
                 noise_dim=100
                 ):
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

        self.vocab_size = len(self.vocab)
        self.batch_size = batch_size
        self.lr = lr
        self.beta1 = 0.5
        self.num_epochs = epochs
        self.num_workers = num_workers

        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=is_cuda)

        self.save_path = save_path
        self.checkpoints_path = os.path.join(self.save_path, 'checkpoints/')
        self.figure_path = os.path.join(self.save_path, 'figures/')
        utils.makedirs(self.checkpoints_path)
        utils.makedirs(self.figure_path)
        utils.makedirs(os.path.join(self.save_path, 'plot/'))

        self.type = type

        # settings for GANs
        self.noise_dim = noise_dim
        self.embed_dim = text_embed_dim
        self.hidden_dim = hidden_dim

        # image G
        self.text_encoder = torch.nn.DataParallel(
            TextEncoder(output_dim=self.hidden_dim,
                        embed_dim=self.embed_dim, hidden_dim=self.hidden_dim,
                        vocab_size=self.vocab_size),
        )
        print('text_encoder=%d' % utils.count_parameters(self.text_encoder))

        self.image_generator64 = torch.nn.DataParallel(
            ImageDecoder64(input_dim=self.hidden_dim),
        )
        print('image_generator64=%d' % utils.count_parameters(self.image_generator64))
        # image D
        self.image_discriminator64 = torch.nn.DataParallel(
            ImageDiscriminator(image_size=64, encoding_dim=self.hidden_dim),
        )
        print('image_discriminator64=%d' % utils.count_parameters(self.image_discriminator64))

        # text G
        self.image_encoder64 = torch.nn.DataParallel(
            ImageEncoder(image_size=64, output_dim=self.hidden_dim),
        )
        print('image_encoder64=%d' % utils.count_parameters(self.image_encoder64))

        self.text_generator = torch.nn.DataParallel(
            TextDecoder(
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                vocab_size=self.vocab_size),
        )
        print('text_generator=%d' % utils.count_parameters(self.text_generator))

        # text D
        self.text_discriminator = torch.nn.DataParallel(
            TextDiscriminator(embed_dim=self.embed_dim,
                              hidden_dim=self.hidden_dim,
                              vocab_size=self.vocab_size),
        )
        print('text_discriminator=%d' % utils.count_parameters(self.text_discriminator))

        # pair (text+image) D
        self.pair_discriminator64 = torch.nn.DataParallel(
            ImageTextPairDiscriminator(
                image_size=64,
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                vocab_size=self.vocab_size),
        )
        print('pair_discriminator64=%d' % utils.count_parameters(self.pair_discriminator64))

        # 2nd stage of image G & D
        self.image_generator128 = torch.nn.DataParallel(
            ImageDecoder128(input_dim=self.hidden_dim),
        )
        print('image_generator128=%d' % utils.count_parameters(self.image_generator128))
        self.image_discriminator128 = torch.nn.DataParallel(
            ImageDiscriminator(image_size=128, encoding_dim=self.hidden_dim),
        )
        print('image_discriminator128=%d' % utils.count_parameters(self.image_discriminator128))
        self.image_encoder128 = torch.nn.DataParallel(
            ImageEncoder(image_size=128, output_dim=self.hidden_dim),
        )
        print('image_encoder128=%d' % utils.count_parameters(self.image_encoder128))
        self.pair_discriminator128 = torch.nn.DataParallel(
            ImageTextPairDiscriminator(
                image_size=128,
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                vocab_size=self.vocab_size),

        )
        print('pair_discriminator128=%d' % utils.count_parameters(self.pair_discriminator128))

        # if is_cuda:
        #     self.image_encoder64 = self.image_encoder64.cuda()
        #     self.text_encoder = self.text_encoder.cuda()
        #     self.image_generator64 = self.image_generator64.cuda()
        #     self.image_discriminator64 = self.image_discriminator64.cuda()
        #
        #     self.text_generator = self.text_generator.cuda()
        #     self.text_discriminator = self.text_discriminator.cuda()
        #     self.pair_discriminator64 = self.pair_discriminator64.cuda()
        #
        #     self.image_encoder128 = self.image_encoder128.cuda()
        #     self.image_generator128 = self.image_generator128.cuda()
        #     self.image_discriminator128 = self.image_discriminator128.cuda()
        #     self.pair_discriminator128 = self.pair_discriminator128.cuda()


        self.optimG_image64 = torch.optim.Adam(itertools.chain(self.text_encoder.parameters(), self.image_generator64.parameters()),
                                               lr=self.lr, betas=(self.beta1, 0.999))
        self.optimD_image64 = torch.optim.Adam(self.image_discriminator64.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.optimD_text = torch.optim.Adam(self.text_discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimD_pair64 = torch.optim.Adam(self.pair_discriminator64.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        G_text_params = itertools.chain(list(self.image_encoder64.parameters()), self.image_encoder128.parameters(), self.text_generator.parameters())
        self.optimG_text = torch.optim.Adam(G_text_params, lr=self.lr * 0.1, betas=(self.beta1, 0.999))

        self.optimG_image128 = torch.optim.Adam(self.image_generator128.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimD_image128 = torch.optim.Adam(self.image_discriminator128.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimD_pair128 = torch.optim.Adam(self.pair_discriminator128.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        # optionally load pretrained checkpoint
        if checkpoint:
            if os.path.isfile(checkpoint):
                print("=> loading checkpoint '{}'".format(checkpoint))

                if is_cuda:
                    checkpoint = torch.load(checkpoint)
                else:
                    checkpoint = torch.load(checkpoint, map_location='cpu')

                self.image_encoder64.load_state_dict(checkpoint['image_encoder64'])
                self.text_encoder.load_state_dict(checkpoint['text_encoder'])
                self.image_generator64.load_state_dict(checkpoint['image_generator64'])
                self.text_generator.load_state_dict(checkpoint['text_generator'])
                self.image_discriminator64.load_state_dict(checkpoint['image_discriminator64'])
                self.text_discriminator.load_state_dict(checkpoint['text_discriminator'])
                self.pair_discriminator64.load_state_dict(checkpoint['pair_discriminator64'])

                self.image_encoder128.load_state_dict(checkpoint['image_encoder128'])
                self.image_generator128.load_state_dict(checkpoint['image_generator128'])
                self.image_discriminator128.load_state_dict(checkpoint['image_discriminator128'])
                self.pair_discriminator128.load_state_dict(checkpoint['pair_discriminator128'])

                # optionally resume from a checkpoint
                if resume:
                    self.start_epoch = checkpoint['epoch']
                    self.optimG_text.load_state_dict(checkpoint['optimG_text'])
                    self.optimD_text.load_state_dict(checkpoint['optimD_text'])
                    self.optimG_image64.load_state_dict(checkpoint['optimG_image64'])
                    self.optimD_image64.load_state_dict(checkpoint['optimD_image64'])
                    self.optimD_pair64.load_state_dict(checkpoint['optimD_pair64'])

                    self.optimG_image128.load_state_dict(checkpoint['optimG_image128'])
                    self.optimD_image128.load_state_dict(checkpoint['optimD_image128'])
                    self.optimD_pair128.load_state_dict(checkpoint['optimD_pair128'])

                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(checkpoint, checkpoint['epoch'] if resume else 0))
            else:
                print("=> no valid checkpoint found at '{}'".format(checkpoint))


    def train(self, cls, interp,
              unimodal_disc, denoised_reconstruct,
              cyclegan, image_stagegan):
        '''

        :param cls:  Matching-aware discriminator (GAN-CLS):
            In addition to the usual real/fake inputs to the discriminator during training,
            a third type of input is added, consisting of real images with mismatched text,
            which the discriminator must learn to score as fake.
        :param interp: Learning with manifold interpolation (GAN-INT):
            Sample an additional text data point, satisfying D on interpolated text embeddings
            and let G learn to fill in gaps on the data manifold in between training points
        :return:
        '''
        if self.type == 'gan':
            self._train_gan(cls, interp)
        elif self.type == 'txt2img':
            self._train_text2image(
                mismatch_cls=cls, interpolation=interp,
                unimodal_disc=unimodal_disc, denoised_reconstruct=denoised_reconstruct,
                cyclegan=cyclegan, image_stagegan=image_stagegan)
        # elif self.type == 'pretrain':
        #     self._pretrain_caption2image()
        elif self.type == 'img2txt':
            self._image2text()


    def _image2text(self):
        # Create model directory
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)
        
        if not os.path.exists(self.figure_path):
            os.makedirs(self.figure_path)

        # Loss and Optimizer (Gen)
        mle_criterion = nn.CrossEntropyLoss(ignore_index=self.vocab('<unk>'))
        # Loss and Optimizer (Disc)
        bce_criterion = nn.BCELoss()

        if is_cuda:
            mle_criterion = mle_criterion.cuda()
            bce_criterion = bce_criterion.cuda()

        gen_pretrain_num_epochs = 0 #50
        disc_pretrain_num_epochs = 0 #20
        gan_num_epochs = 50

        text_gen_losses = []
        pair_disc_losses64 = []
        pair_disc_losses128 = []
        text_disc_losses = []

        num_step_per_epoch = len(self.data_loader)

        for epoch in tqdm(range(max([gen_pretrain_num_epochs, disc_pretrain_num_epochs, gan_num_epochs]))):
            for example_id, example in enumerate(tqdm(self.data_loader)):
            # for sample in self.data_loader:
                right_images64 = example['right_images'] # 64x3x128x128
                right_images128 = example['right_images128'] # 64x3x128x128
                right_captions = example['captions'] # batch_size * max_length
                right_lengths = example['lengths'] # batch_size * 1
                noised_captions = example['noised_captions'] # batch_size * max_length
                noised_lengths = example['noised_lengths'] # batch_size * 1
                wrong_captions = example['wrong_captions'] # batch_size * max_length
                wrong_lengths = example['wrong_lengths'] # batch_size * 1

                # index of BOS in vocab is 1
                init_word = torch.ones((right_images128.size(0), 1)).long()
                txt_noise = torch.randn(1, right_images128.size(0), self.hidden_dim)

                real_labels = torch.ones(right_images128.size(0))
                fake_labels = torch.zeros(right_images128.size(0))
                smoothed_real_labels = torch.FloatTensor(utils.Utils.smooth_label(real_labels.numpy(), -0.1))

                fake_text_length = torch.ones((right_images64.size(0),), dtype=torch.int32)
                fake_text_length.new_full((right_images64.size(0),), 50)

                if is_cuda:
                    right_images64 = right_images64.cuda()
                    right_images128 = right_images128.cuda()
                    right_captions = right_captions.cuda()
                    right_lengths = right_lengths.cuda()
                    noised_captions = noised_captions.cuda()
                    noised_lengths = noised_lengths.cuda()
                    wrong_captions = wrong_captions.cuda()
                    wrong_lengths = wrong_lengths.cuda()
                    real_labels = real_labels.cuda()
                    fake_labels = fake_labels.cuda()
                    smoothed_real_labels = smoothed_real_labels.cuda()
                    init_word = init_word.cuda()
                    txt_noise = txt_noise.cuda()
                    fake_text_length = fake_text_length.cuda()

                # Train generator with teacher forcing
                if epoch < int(gen_pretrain_num_epochs):
                    self.optimG_text.zero_grad()
                    image_encoding64 = self.image_encoder64(right_images64)
                    image_encoding128 = self.image_encoder128(right_images128)
                    # ignore the 1st token (BOS) in target captions, [batch_size, max_len-1]
                    targets = right_captions[:,1:].contiguous().view(-1)
                    # get logits [batch_size, max_len, vocab_size] from decoder
                    #    and truncate the last token (output after EOS is useless)
                    logits64, _ = self.text_generator(image_encoding64, txt_noise, right_captions, right_lengths)
                    logits64 = logits64.view(-1, logits64.shape[-1])
                    loss_gen = mle_criterion(logits64, targets)

                    logits128, _ = self.text_generator(image_encoding128, txt_noise, right_captions, right_lengths)
                    logits128 = logits128.view(-1, logits128.shape[-1])
                    loss_gen += mle_criterion(logits128, targets)

                    loss_gen.backward()
                    self.optimG_text.step()

                    text_gen_losses.append(float(loss_gen.cpu().data.numpy()))

                # Train discriminator with a positive caption and a random negative caption
                if epoch < int(disc_pretrain_num_epochs):
                    # pair and text discriminator
                    self.optimD_pair64.zero_grad()
                    self.optimD_pair128.zero_grad()

                    real_probs64 = self.pair_discriminator64(right_images64.detach(), right_captions.detach(), right_lengths)
                    real_probs128 = self.pair_discriminator128(right_images128.detach(), right_captions.detach(), right_lengths)
                    wrong_probs64 = self.pair_discriminator64(right_images64.detach(), wrong_captions.detach(), wrong_lengths)
                    wrong_probs128 = self.pair_discriminator128(right_images128.detach(), wrong_captions.detach(), wrong_lengths)

                    pair_disc_loss64 = bce_criterion(real_probs64, smoothed_real_labels) + bce_criterion(wrong_probs64, fake_labels)
                    pair_disc_loss128 = bce_criterion(real_probs128, smoothed_real_labels) + bce_criterion(wrong_probs128, fake_labels)

                    pair_disc_loss64.backward()
                    pair_disc_loss128.backward()
                    self.optimD_pair64.step()
                    self.optimD_pair128.step()

                    pair_disc_losses64.append(float(pair_disc_loss64.cpu().data.numpy()))
                    pair_disc_losses128.append(float(pair_disc_loss128.cpu().data.numpy()))

                    # text discriminator
                    self.optimD_text.zero_grad()
                    real_probs = self.text_discriminator(right_captions, right_lengths)
                    wrong_probs = self.text_discriminator(noised_captions, noised_lengths)

                    real_loss = bce_criterion(real_probs, smoothed_real_labels)
                    wrong_loss = bce_criterion(wrong_probs, fake_labels)
                    text_disc_loss = real_loss + wrong_loss # + fake_loss, no fake_loss because this is pretraining

                    text_disc_loss.backward()
                    self.optimD_text.step()
                    text_disc_losses.append(float(text_disc_loss.cpu().data.numpy()))

                # Train text GAN
                if epoch < gan_num_epochs:
                    self.optimD_pair64.zero_grad()
                    self.optimD_pair128.zero_grad()
                    self.optimD_text.zero_grad()
                    self.optimG_text.zero_grad()

                    image_encoding64 = self.image_encoder64(right_images64)
                    image_encoding128 = self.image_encoder128(right_images128)

                    # Train D
                    # Fake text [batch_size, max_len]
                    fake_texts64 = self.text_generator.module.sample(image_encoding64, txt_noise, init_word, max_len=50)
                    fake_texts128 = self.text_generator.module.sample(image_encoding128, txt_noise, init_word, max_len=50)

                    # train D once while train G 5 times
                    if example_id % 5 == 0:
                        pair_real_probs64 = self.pair_discriminator64(right_images64.detach(), right_captions.detach(), lengths=right_lengths)
                        pair_real_probs128 = self.pair_discriminator128(right_images128.detach(), right_captions.detach(), lengths=right_lengths)
                        text_real_probs = self.text_discriminator(right_captions.detach(), lengths=right_lengths)

                        pair_fake_probs64 = self.pair_discriminator64(right_images64.detach(), fake_texts64.detach(), lengths=fake_text_length)
                        pair_fake_probs128 = self.pair_discriminator128(right_images128.detach(), fake_texts128.detach(), lengths=fake_text_length)
                        text_fake_probs = self.text_discriminator(fake_texts64.detach(), lengths=fake_text_length)

                        pair_disc_loss64 = bce_criterion(pair_real_probs64, smoothed_real_labels) + \
                                            bce_criterion(pair_fake_probs64, fake_labels)
                        pair_disc_loss128 = bce_criterion(pair_real_probs128, smoothed_real_labels) + \
                                            bce_criterion(pair_fake_probs128, fake_labels)
                        text_disc_loss = bce_criterion(text_real_probs, smoothed_real_labels) + \
                                         bce_criterion(text_fake_probs, fake_labels)

                        pair_disc_loss64.backward()
                        pair_disc_loss128.backward()
                        text_disc_loss.backward()

                        self.optimD_pair64.step()
                        self.optimD_pair128.step()
                        self.optimD_text.step()

                        pair_disc_losses64.append(float(pair_disc_loss64.cpu().data.numpy()))
                        pair_disc_losses128.append(float(pair_disc_loss128.cpu().data.numpy()))
                        text_disc_losses.append(float(text_disc_loss.cpu().data.numpy()))

                    # Train G
                    self.optimG_text.zero_grad()
                    pair_fake_probs64 = self.pair_discriminator64(right_images64, fake_texts64, lengths=fake_text_length)
                    pair_fake_probs128 = self.pair_discriminator128(right_images128, fake_texts128, lengths=fake_text_length)
                    text_fake_probs = self.text_discriminator(fake_texts64, lengths=fake_text_length)

                    pair_disc_loss64 = bce_criterion(pair_fake_probs64, real_labels)
                    pair_disc_loss128 = bce_criterion(pair_fake_probs128, real_labels)
                    text_disc_loss = bce_criterion(text_fake_probs, real_labels)

                    loss_gen = pair_disc_loss64 + pair_disc_loss128 + text_disc_loss
                    loss_gen.backward()
                    self.optimG_text.step()

                    text_gen_losses.append(float(loss_gen.cpu().data.numpy()))

            if (epoch + 1) % 1 == 0:
                # Save checkpoint
                print('Saving checkpoint to %s' % (os.path.join(self.checkpoints_path, 'pretrained-img2txt-%d.pkl' % (epoch + 1))))
                utils.save_checkpoint({**self.state_to_dict(), **{'epoch': epoch}}, is_best=False,
                                      filepath=os.path.join(self.checkpoints_path, 'pretrained-img2txt-%d.pkl' % (epoch + 1)))
                # Plot pretraining figures
                print('Saving learning curve plot to %s' % (self.figure_path + 'pretraining_img2txt_learning_curve.png'))
                plt.plot(text_gen_losses, 'o-', color="r", label='text_gen_loss')
                plt.plot(pair_disc_losses64, 'o-', color="g", label='pair_disc_loss64')
                plt.plot(pair_disc_losses128, 'o-', color="y", label='pair_disc_loss128')
                plt.plot(text_disc_losses, 'o-', color="b", label='text_disc_loss')
                plt.title('Pretraining img2txt Learning Curve')
                plt.grid()
                plt.legend(loc="best")
                plt.savefig(self.figure_path + 'pretraining_img2txt_learning_curve.png')
                plt.clf()

                # Save loss to file
                with open(self.figure_path + 'pretraining_img2txt_loss.csv', 'w') as loss_csv:
                    loss_csv.write('epoch, step, gen, pair_disc64, pair_disc128, text_disc\n')
                    for step_i in range(max(len(text_gen_losses), len(pair_disc_losses64), len(pair_disc_losses128), len(text_disc_losses))):
                        line = '%d, %d, ' % (step_i / num_step_per_epoch + 1, step_i)
                        for loss in [text_gen_losses, pair_disc_losses64, pair_disc_losses128, text_disc_losses]:
                            if step_i < len(loss):
                                line += '%.6f, ' % loss[step_i]
                            else:
                                line += '0.0, '
                        line += '\n'
                        loss_csv.write(line)

        # Save pretrained models
        utils.save_checkpoint({**self.state_to_dict(), **{'epoch': epoch}}, is_best=False,
                              filepath=os.path.join(self.checkpoints_path, 'pretrained-img2txt-%d-final.pkl' % (epoch + 1)))


    def state_to_dict(self, names=None):
        if not names:
            names = ['image_encoder64', 'image_encoder128',
                     'text_encoder',
                     'image_generator64', 'image_discriminator64',
                     'image_generator128', 'image_discriminator128',
                     'text_generator', 'text_discriminator',
                     'pair_discriminator64', 'pair_discriminator128',
                     'optimG_text', 'optimD_text',
                     'optimG_image64', 'optimG_image128',
                     'optimD_image64', 'optimD_image128',
                     'optimD_pair64', 'optimD_pair128']

        return {name: getattr(self, name).state_dict() for name in names}


    def _train_text2image(self, mismatch_cls=True, interpolation=True,
                          unimodal_disc=True, denoised_reconstruct=True,
                          cyclegan=False, image_stagegan=False):
        exp_name = 'stack_gan'
        if mismatch_cls:
            exp_name += '_cls'
        if interpolation:
            exp_name += '_int'
        if unimodal_disc:
            exp_name += '_unimodal'
        if denoised_reconstruct:
            exp_name += '_denoise'
        if cyclegan:
            exp_name += '_cycle'
        if image_stagegan:
            exp_name += '_stage'

        iteration = 0
        num_step_per_epoch = len(self.data_loader)

        bce_criterion = nn.BCELoss()
        l2_criterion = nn.MSELoss()
        l1_criterion = nn.L1Loss()
        mle_criterion = nn.CrossEntropyLoss()

        if is_cuda:
            bce_criterion = bce_criterion.cuda()
            l2_criterion = l2_criterion.cuda()
            l1_criterion = l1_criterion.cuda()
            mle_criterion = mle_criterion.cuda()

        # cycle gan params
        lambda_a = 2
        lambda_b = 2

        pair64_disc_losses = []
        img64_disc_losses = []
        img64_disc_real_losses = []
        img64_disc_fake_losses = []
        img64_gen_losses = []
        img64_gen_unimodal_losses = []
        img64_gen_denoised_losses = []
        img64_gen_interp_losses = []
        img64_gen_cycle_losses = []
        img64_gen_cycle_TIT_losses = []
        img64_gen_cycle_ITI_losses = []


        for epoch in tqdm(range(self.num_epochs)):
            for example_id, example in enumerate(tqdm(self.data_loader)):
                start_time = time.time()

                iteration += 1
                right_images64 = example['right_images'] # bsx3x64x64
                wrong_images64 = example['wrong_images'] # bsx3x64x64
                right_images128 = example['right_images128'] # bsx3x128x128
                wrong_images128 = example['wrong_images128'] # bsx3x128x128
                right_captions = example['captions']
                right_lengths = example['lengths']
                wrong_captions = example['wrong_captions']
                wrong_lengths = example['wrong_lengths']

                init_word = torch.ones((right_images64.size(0), 1)).long()
                text_noise = torch.randn(1, right_images64.size(0), self.hidden_dim)

                real_labels = Variable(torch.ones(right_images64.size(0)))
                fake_labels = Variable(torch.zeros(right_images64.size(0)))

                # ======== One sided label smoothing ==========
                # Helps preventing the discriminator from overpowering the
                # generator adding penalty when the discriminator is too confident
                # =============================================
                smoothed_real_labels = Variable(torch.FloatTensor(utils.Utils.smooth_label(real_labels.numpy(), -0.1)))

                fake_text_length = torch.ones((right_images64.size(0),), dtype=torch.int32)
                fake_text_length.new_full((right_images64.size(0),), 50)

                if is_cuda:
                    right_images64 = right_images64.cuda()
                    wrong_images64 = wrong_images64.cuda()
                    right_images128 = right_images128.cuda()
                    wrong_images128 = wrong_images128.cuda()
                    right_captions = right_captions.cuda()
                    right_lengths = right_lengths.cuda()
                    wrong_captions = wrong_captions.cuda()
                    wrong_lengths = wrong_lengths.cuda()

                    real_labels = real_labels.cuda()
                    smoothed_real_labels = smoothed_real_labels.cuda()
                    fake_labels = fake_labels.cuda()
                    init_word = init_word.cuda()
                    text_noise = text_noise.cuda()
                    fake_text_length = fake_text_length.cuda()

                noised_images64 = utils.add_gaussion_noise(right_images64, cuda=is_cuda)

                # ------------------- Training -------------------------------
                # ------------------- Encoding text -------------------------------
                text_encoding = self.text_encoder(right_captions, right_lengths)
                ##########################################################
                # Training D stage 1: maximize log(D(x)) + log(1 - D(G(z)))
                ##########################################################
                # (image, text) pair D
                self.optimD_pair64.zero_grad()

                pair_disc64_outputs = self.pair_discriminator64(
                    right_images64.detach(),
                    right_captions.detach(),
                    lengths=right_lengths
                )
                pair_disc64_real_loss = bce_criterion(pair_disc64_outputs, smoothed_real_labels)

                # obtain a fake image
                image_noise = Variable(torch.randn(right_images64.size(0), self.noise_dim))
                image_noise = image_noise.view(image_noise.size(0), self.noise_dim, 1, 1)
                if is_cuda:
                    image_noise = image_noise.cuda()

                fake_images64 = self.image_generator64(text_encoding, image_noise)
                pair_disc64_outputs = self.pair_discriminator64(
                    fake_images64.detach(),
                    right_captions.detach(),
                    lengths=right_lengths
                )
                pair_disc64_fake_loss = bce_criterion(pair_disc64_outputs, fake_labels)

                if mismatch_cls:
                    pair_disc64_outputs = self.pair_discriminator64(
                        wrong_images64.detach(),
                        right_captions.detach(),
                        lengths=right_lengths
                    )
                    pair_disc64_wrong_loss = bce_criterion(pair_disc64_outputs, fake_labels)
                    pair64_disc_loss = pair_disc64_real_loss + 0.5 * pair_disc64_wrong_loss + 0.5 * pair_disc64_fake_loss
                else:
                    pair64_disc_loss = pair_disc64_real_loss + pair_disc64_fake_loss

                pair64_disc_loss.backward()
                self.optimD_pair64.step()
                pair64_disc_losses.append(float(pair64_disc_loss.cpu().data.numpy()))

                # image D
                if unimodal_disc:
                    # self.optimD_image64.zero_grad()
                    self.image_discriminator64.zero_grad()
                    image_disc64_fake_outputs = self.image_discriminator64(fake_images64.detach())
                    img_disc64_fake_loss = bce_criterion(image_disc64_fake_outputs, fake_labels)
                    # right_images64 is too easy? change to noised_image64
                    # image_disc64_real_outputs = self.image_discriminator64(noised_image64.detach())
                    # img_disc64_real_loss = bce_criterion(image_disc64_real_outputs, smoothed_real_labels)
                    image_disc64_real_outputs = self.image_discriminator64(right_images64.detach())
                    img_disc64_real_loss = bce_criterion(image_disc64_real_outputs, real_labels)

                    img64_disc_loss = img_disc64_real_loss + img_disc64_fake_loss
                    '''
                    # a noised sample might not be necessary?
                    noised_image64 = utils.add_gaussion_noise(right_images64, cuda=is_cuda)
                    img_disc64_outputs, _ = self.image_discriminator64(noised_image64)
                    img_disc64_wrong_loss = criterion(img_disc64_outputs, fake_labels)

                    if mismatch_cls:
                        img64_disc_loss = img_disc64_real_loss + 0.5 * img_disc64_wrong_loss + 0.5 * img_disc64_fake_loss
                    else:
                        img64_disc_loss = img_disc64_real_loss + img_disc64_fake_loss
                    '''
                    img64_disc_loss.backward()
                    self.optimD_image64.step()
                    img64_disc_losses.append(float(img64_disc_loss.cpu().data.numpy()))
                    img64_disc_real_losses.append(float(img_disc64_real_loss.cpu().data.numpy()))
                    img64_disc_fake_losses.append(float(img_disc64_fake_loss.cpu().data.numpy()))
                    print('\nimg_disc64_real_loss=%.4f, img_disc64_fake_loss=%.4f, img_disc64_loss=%.4f' % (img_disc64_real_loss, img_disc64_fake_loss, img64_disc_loss))

                ##########################################################
                # Training G stage 1: maximize log(D(G(z)))
                ##########################################################
                self.optimG_image64.zero_grad()
                self.optimD_image64.zero_grad()
                self.optimD_pair64.zero_grad()
                self.optimD_pair128.zero_grad()
                # if is_cuda:
                #     noise = Variable(torch.randn(right_images64.size(0), self.noise_dim)).cuda()
                # else:
                #     noise = Variable(torch.randn(right_images64.size(0), self.noise_dim))
                # noise = noise.view(noise.size(0), self.noise_dim, 1, 1)
                # fake_images = self.image_generator64(right_embed, noise)
                pair64_disc_outputs = self.pair_discriminator64(fake_images64, right_captions, right_lengths)
                img64_gen_loss = bce_criterion(pair64_disc_outputs, real_labels)

                if unimodal_disc:
                    img64_disc_outputs = self.image_discriminator64(fake_images64)
                    img64_unimodal_gen_loss = bce_criterion(img64_disc_outputs, real_labels)
                    print('gen_image_disc64_loss=%.4f' % img64_unimodal_gen_loss)
                    img64_gen_loss += 2.0 * img64_unimodal_gen_loss

                    img64_gen_unimodal_losses.append(2.0 * float(img64_unimodal_gen_loss.cpu().data.numpy()))

                if denoised_reconstruct:
                    img64_encoding = self.image_encoder64(noised_images64)
                    reconstrcted_image64  = self.image_generator64(img64_encoding, image_noise)
                    img64_l1_loss = l1_criterion(reconstrcted_image64, right_images64)
                    img64_l2_loss = l2_criterion(reconstrcted_image64, right_images64)
                    img64_gen_loss += 0.8 * img64_l1_loss + 0.2 * img64_l2_loss

                    img64_gen_denoised_losses.append(0.8 * float(img64_l1_loss.cpu().data.numpy())
                                                     + 0.2 * float(img64_l2_loss.cpu().data.numpy()))

                if (interpolation):
                    """ GAN INT loss, let G generate with an interpolated text to fill the text space"""
                    available_batch_size = int(text_encoding.size(0))
                    # obtain an interpolation of text embedding by merging the two halves in batch
                    # [batch_size/2, max_len, embed_dim]
                    first_part = text_encoding[: int(available_batch_size / 2), :]
                    second_part = text_encoding[int(available_batch_size / 2): , :]
                    interp_embed = (first_part + second_part) * 0.5

                    # noise.shape = [batch_size/2, self.noise_dim]
                    if is_cuda:
                        image_noise = Variable(torch.randn(int(available_batch_size/2), self.noise_dim)).cuda()
                    else:
                        image_noise = Variable(torch.randn(int(available_batch_size/2), self.noise_dim))

                    image_noise = image_noise.view(image_noise.size(0), self.noise_dim, 1, 1)

                    # [batch_size / 2]
                    interp_real_labels = Variable(torch.ones(int(available_batch_size/2)))
                    if is_cuda:
                        interp_real_labels = interp_real_labels.cuda()

                    fake_images64 = self.image_generator64(interp_embed, image_noise)
                    pair64_disc_outputs = self.pair_discriminator64.module.forward_interpolate(fake_images64, interp_embed)
                    g_pair_int_loss = bce_criterion(pair64_disc_outputs, interp_real_labels)
                    img64_gen_loss += 0.5 * g_pair_int_loss
                    int_loss = 0.5 * float(g_pair_int_loss.cpu().data.numpy())

                    # [batch_size/2,nc,64,64,64]
                    if unimodal_disc:
                        fake_images64 = self.image_generator64(interp_embed, image_noise)
                        img64_disc_outputs = self.image_discriminator64(fake_images64)
                        g_image_int_loss = bce_criterion(img64_disc_outputs, interp_real_labels)
                        img64_gen_loss += 0.5 * g_image_int_loss

                        int_loss += 0.5 * float(g_image_int_loss.cpu().data.numpy())

                    img64_gen_interp_losses.append(int_loss)

                img64_gen_loss.backward()
                self.optimG_image64.step()
                img64_gen_losses.append(float(img64_gen_loss.cpu().data.numpy()))

                # -------------------- Training G with Cycle Consistency -------------------------------
                if cyclegan:
                    self.optimG_image64.zero_grad()
                    self.optimG_text.zero_grad()

                    text_noise = torch.randn(1, right_images64.size(0), self.hidden_dim)
                    image_noise = torch.randn(right_images64.size(0), self.noise_dim)
                    image_noise = image_noise.view(image_noise.size(0), self.noise_dim, 1, 1)
                    if is_cuda:
                        text_noise = text_noise.cuda()
                        image_noise = image_noise.cuda()

                    # text -> image -> text
                    text_encoding64 = self.text_encoder(right_captions, text_lengths=right_lengths)
                    gen_image = self.image_generator64(text_encoding64, z=image_noise)
                    gen_image_encoding64 = self.image_encoder64(gen_image)
                    recovered_text = self.text_generator.module.sample(gen_image_encoding64, text_noise, init_word, max_len=50)
                    recovered_text = recovered_text[:, :right_captions.size(1)].long()

                    cycle_TIT_outputs = self.text_discriminator(recovered_text, lengths=fake_text_length)
                    loss_cycle_TIT = bce_criterion(cycle_TIT_outputs, real_labels) * 10.0

                    # image -> text -> image
                    image_encoding64 = self.image_encoder64(right_images64)
                    gen_text = self.text_generator.module.sample(image_encoding64, text_noise, init_word, max_len=50)
                    gen_text_encoding64 = self.text_encoder(gen_text, text_lengths=fake_text_length)
                    recovered_image = self.image_generator64(gen_text_encoding64, z=image_noise)

                    loss_cycle_ITI = l2_criterion(right_images64, recovered_image) * 10.0

                    cycle_loss_G = loss_cycle_TIT + loss_cycle_ITI
                    cycle_loss_G.backward()
                    self.optimG_image64.step()
                    self.optimG_text.step()

                    img64_gen_cycle_losses.append(float(cycle_loss_G.cpu().data.numpy()))
                    img64_gen_cycle_TIT_losses.append(float(loss_cycle_TIT.cpu().data.numpy()))
                    img64_gen_cycle_ITI_losses.append(float(loss_cycle_ITI.cpu().data.numpy()))

                if image_stagegan:
                    # -------------------- Training D stage 2 -------------------------------
                    self.image_discriminator128.zero_grad()
                    outputs = self.image_discriminator128(right_images128, text_encoding)
                    real_loss = bce_criterion(outputs, smoothed_real_labels)
                    real_score = outputs

                    if mismatch_cls:
                        outputs = self.image_discriminator128(wrong_images128, text_encoding)
                        wrong_loss = bce_criterion(outputs, fake_labels)
                        wrong_score = outputs

                    image_noise = torch.randn(right_images64.size(0), self.noise_dim).view(image_noise.size(0), self.noise_dim, 1, 1)
                    if is_cuda:
                        image_noise = image_noise.cuda()

                    fake_images_v1 = self.image_generator64(text_encoding, image_noise)
                    fake_images_v1 = fake_images_v1.detach()
                    fake_images64 = self.generator128(fake_images_v1, text_encoding)
                    fake_images64 = fake_images64.detach()
                    outputs = self.image_discriminator128(fake_images64, text_encoding)
                    fake_loss = bce_criterion(outputs, fake_labels)
                    fake_score = outputs

                    if mismatch_cls:
                        d_loss2 = real_loss + 0.5 * wrong_loss + 0.5 * fake_loss
                    else:
                        d_loss2 = real_loss + fake_loss

                    d_loss2.backward()
                    self.optimD_image128.step()

                    # -------------------- Training G stage 2 -------------------------------
                    self.generator128.zero_grad()
                    self.image_discriminator128.zero_grad()
                    if is_cuda:
                        image_noise = Variable(torch.randn(right_images64.size(0), self.noise_dim)).cuda()
                    else:
                        image_noise = Variable(torch.randn(right_images64.size(0), self.noise_dim))

                    image_noise = image_noise.view(image_noise.size(0), self.noise_dim, 1, 1)
                    fake_images_v1 = self.image_generator64(text_encoding, image_noise)
                    fake_images_v1 = fake_images_v1.detach()
                    fake_images64 = self.generator128(fake_images_v1, text_encoding)
                    outputs = self.image_discriminator128(fake_images64, text_encoding)

                    g_loss2 = bce_criterion(outputs, real_labels)
                    g_loss2.backward()
                    self.optimG_image128.step()

                    gen_losses.append(g_loss2.data[0])
                    disc_losses.append(d_loss2.data[0])

                    # Generate caption with caption GAN (inverse GAN)
                    # fake_images.requires_grad = False # freeze the caption generator
                    self.text_generator.zero_grad()
                    sampled_captions, _ = self.text_generator.forward(fake_images64, right_captions, right_lengths)
                    targets = pack_padded_sequence(right_captions, right_lengths, batch_first=True)[0]
                    loss_cycle_A = mle_criterion(sampled_captions, targets)* lambda_a
                    loss_cycle_A.backward()
                    self.optimG_image128.step()
                    self.optimG_text.step()
                    cycle_a_losses.append(loss_cycle_A.data[0])

                end_time = time.time()
                print('Training a batch: %f seconds' % (end_time - start_time))

            if (epoch + 1) % 1 == 0:
                # Save checkpoint
                print('Saving checkpoint to %s' % (os.path.join(self.checkpoints_path, '%s-%d.pkl' % (exp_name, epoch + 1))))
                utils.save_checkpoint({**self.state_to_dict(), **{'epoch': epoch}}, is_best=False,
                                      filepath=os.path.join(self.checkpoints_path, '%s-%d.pkl' % (exp_name, epoch + 1)))
                # Plot pretraining figures
                print('Saving learning curve plot to %s' % (self.figure_path + '%s learning_curve.png' % (exp_name)))
                plt.plot(pair64_disc_losses, '.-', color="r", label='pair64_disc_losses', linewidth=0.1)
                plt.plot(img64_disc_losses, '.-', color="g", label='img64_disc_losses', linewidth=0.1)
                plt.plot(img64_gen_losses, '.-', color="y", label='img64_gen_losses', linewidth=0.1)
                plt.plot(img64_disc_real_losses, '.-', color="blue", label='img64_disc_real_losses', linewidth=0.1)
                plt.plot(img64_disc_fake_losses, '.-', color="pink", label='img64_disc_fake_losses', linewidth=0.1)
                plt.plot(img64_gen_unimodal_losses, '.-', color="cyan", label='img64_gen_unimodal_losses', linewidth=0.1)
                plt.plot(img64_gen_denoised_losses, '.-', color="magenta", label='img64_gen_denoised_losses', linewidth=0.1)
                plt.plot(img64_gen_interp_losses, '.-', color="black", label='img64_gen_interp_losses', linewidth=0.1)

                # plt.plot(img64_gen_cycle_losses, 'o-', color="cyan", label='img64_gen_cycle_losses', linewidth=0.2)
                # plt.plot(img64_gen_cycle_TIT_losses, '.-', color="magenta", label='img64_gen_cycle_TIT_losses', linewidth=0.1)
                # plt.plot(img64_gen_cycle_ITI_losses, '.-', color="black", label='img64_gen_cycle_ITI_losses', linewidth=0.1)
                plt.title('%s Learning Curve' % (exp_name))
                plt.grid()
                plt.legend(loc="best")
                plt.savefig(self.figure_path + '%s_learning_curve.png' % (exp_name))
                plt.clf()

                # Save loss to file
                with open(self.figure_path + '%s_loss.csv' % (exp_name), 'w') as loss_csv:
                    loss_csv.write('epoch, step, gen, '
                                   'img_gen64, pair_disc64, '
                                   'img_disc64, img_disc_real64, img_disc_fake64, '
                                   'gen_cycle64, gen_cycle_TIT64, gen_cycle_ITI64, '
                                   '\n')
                    for step_i in range(max(len(pair64_disc_losses),
                                            len(img64_disc_losses),
                                            len(img64_disc_losses),
                                            len(img64_disc_real_losses),
                                            len(img64_disc_fake_losses))):
                        line = '%d, %d, ' % (step_i / num_step_per_epoch + 1, step_i)
                        for loss in [img64_gen_losses, pair64_disc_losses, img64_disc_losses,
                                     img64_disc_real_losses, img64_disc_fake_losses,
                                     img64_gen_cycle_losses, img64_gen_cycle_TIT_losses, img64_gen_cycle_ITI_losses]:
                            if step_i < len(loss):
                                line += '%.6f, ' % loss[step_i]
                            else:
                                line += '0.0, '
                        line += '\n'
                        loss_csv.write(line)

                # plot images in the last batch
                for b_i in range(fake_images64.size(0)):
                    im = Image.fromarray(right_images64.data[b_i].squeeze(0).mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                    im.save('{0}/plot/{1}-GT_64.jpg'.format(self.save_path, b_i))

                    im = Image.fromarray(fake_images64.data[b_i].squeeze(0).mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                    im.save('{0}/plot/{1}-GEN64.jpg'.format(self.save_path, b_i))


    def predict(self, gan_type='gan'):
        torch.manual_seed(7)
        count = 0
        for batch_id, batch in enumerate(tqdm(self.data_loader)):
            right_images = batch['right_images']
            right_images128 = batch['right_images128']
            right_images_original = batch['right_images_original']
            right_captions = batch['captions']
            right_lengths = batch['lengths']
            txt = batch['txt']
            # index of BOS in vocab is 1
            init_word = torch.ones((len(txt), 1)).long()

            if not os.path.exists('results/{0}'.format(self.save_path)):
                os.makedirs('results/{0}'.format(self.save_path))

            img_noise = torch.randn(right_images.size(0), self.noise_dim)
            img_noise = img_noise.view(img_noise.size(0), self.noise_dim, 1, 1)
            txt_noise = torch.randn(1, right_images.size(0), self.hidden_dim)

            if is_cuda:
                right_images = right_images.cuda()
                right_captions = right_captions.cuda()
                right_lengths = right_lengths.cuda()
                img_noise = img_noise.cuda()
                txt_noise = txt_noise.cuda()

            right_embed = self.text_encoder(right_captions, right_lengths)
            fake_images64 = self.image_generator64(right_embed, img_noise)

            # Generate images given captions
            image_encoding64 = self.image_encoder64(right_images)
            image_encoding128 = self.image_encoder128(right_images128)
            # [batch_size, max_len]
            fake_captions64 = self.text_generator.module.sample(image_encoding64, txt_noise, init_word, max_len=50)
            fake_captions128 = self.text_generator.module.sample(image_encoding128, txt_noise, init_word, max_len=50)

            if (gan_type=='stackgan'):
                fake_images_B = self.generator128(fake_images64, right_embed)

            caption_64 = fake_captions64.data.cpu().numpy()
            caption_128 = fake_captions128.data.cpu().numpy()

            for example_i in range(len(txt)):
                example_id = batch_id * self.batch_size + example_i

                gen_image_64 = fake_images64[example_i]
                right_image_128 = right_images128[example_i]
                right_image_original = right_images_original[example_i]
                # truncate pad
                try:
                    end_index = caption_64[example_i].tolist().index(2)
                except ValueError:
                    try:
                        end_index = caption_64[example_i].tolist().index(0)
                    except ValueError:
                        end_index = len(caption_64[example_i])

                caption_64_tokens = self.vocab.decode(caption_64[example_i][:end_index])
                gen64_text = ' '.join(caption_64_tokens)[:200]

                try:
                    end_index = caption_128[example_i].tolist().index(2)
                except ValueError:
                    try:
                        end_index = caption_128[example_i].tolist().index(0)
                    except ValueError:
                        end_index = len(caption_128[example_i])

                caption_128_tokens = self.vocab.decode(caption_128[example_i][:end_index])
                gen128_text = ' '.join(caption_128_tokens)[:200]
                gt_text = txt[example_i].strip().replace("/", "")[:200]

                if not os.path.exists('%s/predict/' % self.save_path):
                    os.makedirs('%s/predict/' % self.save_path)

                im = Image.fromarray(right_image_128.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                im.save('{0}/predict/[{1}][GT128]{2}.jpg'.format(self.save_path, example_id, gt_text))

                im = right_image_original
                im.save('{0}/predict/[{1}][Origin]{2}.jpg'.format(self.save_path, example_id, gt_text))

                im = Image.fromarray(gen_image_64.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                im.save('{0}/predict/[{1}][Gen64]{2}.jpg'.format(self.save_path, example_id, gen64_text))

                '''
                if (gan_type == 'stackgan'):
                    image_128 = fake_images_128[example_id]
                    im = Image.fromarray(image_128.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                    im.save('{0}/predict/[{1}][GEN128]{2}.jpg'.format(self.save_path, example_id, gen128_text))
                '''

                print('-=' * 20)
                print('GT-text : ' + gt_text)
                print('GEN64-text: ' + gen64_text)
                print('GEN128-text: ' + gen128_text)

            count += 1
            # if count == 1:
            #     break






