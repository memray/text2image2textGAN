import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import *
from torch.autograd import Variable
import pdb
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from skipthoughts import UniSkip

is_cuda = torch.cuda.is_available()


class Image2TextGenerator(nn.Module):
    """
    Not used now
    Generate texts (captions) for a set of images
    """
    def __init__(self, bridge_dim, embed_dim, vocab_size, num_layers, initial_noise=True,
                 encoder_model_name='resnet50', decoder_model_name='lstm'):
        super(Image2TextGenerator, self).__init__()
        self.encoder = ImageEncoder(bridge_dim, encoder_model_name=encoder_model_name)
        self.decoder = TextDecoder(embed_dim=embed_dim,
                                   hidden_dim=bridge_dim,
                                   vocab_size=vocab_size,
                                   num_layers=num_layers,
                                   decoder_model_name=decoder_model_name)
        self.features = None
        self.initial_noise = initial_noise

    def forward(self, images, captions, lengths):
        """
        Teacher forcing for training caption generation
        """
        self.features = self.encoder(images) # batch_size * hidden_dim
        logits, lengths = self.decoder(self.features, captions, lengths, initialize_noise=self.initial_noise)
        return logits, lengths

    def pre_compute(self, gen_samples, t):
        """
        pre compute the most likely vocabs and their states
        """
        if self.features is None:
            print('must do forward before calling this function')
            return None

        predicted_ids, saved_states = self.decoder.pre_compute(self.features, gen_samples, t)
        return predicted_ids, saved_states

    def rollout(self, gen_samples, t, saved_states):
        """ inputs:
                * gen_samples: (b, Tmax)
                * t: scalar

            outputs:
                * gen_rollouts: (b, Tmax - t)
                * lengths_rollouts: list (b)
        """
        if self.features is None:
            print('must do forward before calling this function')
            return None

        Tmax = gen_samples.size(1)
        sampled_ids = self.decoder.rollout(self.features, gen_samples, t, Tmax, states=saved_states)
        # pdb.set_trace()
        return sampled_ids

    def sample(self, images, states=None):
        features = self.encoder(images)
        return self.decoder.sample(features, states)


