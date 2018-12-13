import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import *

from torch.nn.functional import gumbel_softmax

is_cuda = torch.cuda.is_available()


class Concat_embed(nn.Module):
    '''
    Concatenating two embeddings
    During forward pass, first map
    '''
    def __init__(self, embed_dim, projected_embed_dim):
        super(Concat_embed, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=projected_embed_dim),
            nn.BatchNorm1d(num_features=projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, inp, embed):
        '''
        Reshape embed from shape of [bs, embed_dim] to shape of [bs, projected_embed_dim]
        Then concatenate inp and reshaped embed
        :param inp:
        :param embed:
        :return:
        '''
        projected_embed = self.projection(embed)
        replicated_embed = projected_embed.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
        hidden_concat = torch.cat([inp, replicated_embed], 1)

        return hidden_concat


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class ImageDecoder64(nn.Module):
    '''
    output is (nc) x 64 x 64
    '''
    def __init__(self, input_dim=128):
        super(ImageDecoder64, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.noise_dim = 100
        self.embed_dim = input_dim
        self.projected_embed_dim = 128
        self.latent_dim = self.noise_dim + self.projected_embed_dim
        # self.ngf = 64
        self.ngf = 16

        self.projection = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
            nn.BatchNorm1d(num_features=self.projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # layers marked with * based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
        self.netG = nn.Sequential(
            # *
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),

            # * adding extra convs will give output (ngf*8) x 4 x 4
            nn.Conv2d(self.ngf * 8, self.ngf * 2, 1, 1, 0),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),

            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, 1, 1),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),

            nn.Conv2d(self.ngf * 2, self.ngf * 8, 3, 1, 1),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),

            # * state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),

            # adding extra convs will give output (ngf*4) x 4 x 4
            nn.Conv2d(self.ngf * 4, self.ngf, 1, 1, 0),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            nn.Conv2d(self.ngf, self.ngf, 3, 1, 1),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            nn.Conv2d(self.ngf, self.ngf * 4, 3, 1, 1),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),

            # * state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # * state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # * state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=True),
            nn.Tanh()
            # state size. (num_channels) x 64 x 64
        )

    def forward(self, embed_vector, z):

        # embed_vector = 64 by 1024
        # projected_embed = 64 by 128 by 1 by 1
        # z = 64 by 100 by 1 by 1

        projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(3)
        latent_vector = torch.cat([projected_embed, z], 1)
        output = self.netG(latent_vector)

        return output


class ImageDecoder128(nn.Module):
    '''
    output is (nc) x 128 x 128
    '''
    def __init__(self, input_dim=128):
        super(ImageDecoder128, self).__init__()
        self.image_size = 128
        self.num_channels = 3
        self.noise_dim = 100
        self.embed_dim = input_dim
        self.projected_embed_dim = 128
        self.latent_dim = self.noise_dim + self.projected_embed_dim
        # self.ngf = 64
        self.ngf = 16

        self.projection = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
            nn.BatchNorm1d(num_features=self.projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # downsample
        self.netG_1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.num_channels, self.ngf, 3, 1, 1),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.Conv2d(self.ngf, self.ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True)
            # state size. (ngf*4) x 16 x 16
        )

        self.join_embed = nn.Sequential(

            nn.Conv2d(self.projected_embed_dim + self.ngf*4 , self.ngf * 4, 3, 1, 1),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True)
        )

        self.residual = self._make_layer(ResBlock, self.ngf * 4, 4)

        # --> 2ngf x 32 x 32
        self.upsample1 = upBlock(self.ngf * 4, self.ngf * 2)
        # --> ngf x 64 x 64
        self.upsample2 = upBlock(self.ngf * 2, self.ngf)
        # --> ngf // 2 x 128 x 128
        self.upsample3 = upBlock(self.ngf, self.ngf // 2)
        # --> ngf // 4 x 128 x 128
        self.flatten = nn.Conv2d(self.ngf // 2 , self.ngf //4, 3, 1, 1)
        # --> 3 x 128 x 128
        self.img = nn.Sequential(
            nn.Conv2d(self.ngf // 4 , 3, 3, 1, 1),
            nn.Tanh()
        )

    def _make_layer(self, block, channel_num, r_num):
        layers = []
        for i in range(r_num):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)


    def forward(self, inp, embed):

        # embed_vector = 64 by 1024
        # projected_embed = 64 by 128 by 1 by 1
        # z = 64 by 100 by 1 by 1
        g1_output = self.netG_1(inp) # shape is (ngf*4) x 16 x 16
        projected_embed = self.projection(embed)
        replicated_embed = projected_embed.repeat(16, 16, 1, 1).permute(2, 3, 0, 1)
        hidden_concat = torch.cat([g1_output, replicated_embed], 1)
        x = self.join_embed(hidden_concat)
        x = self.residual(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.flatten(x)
        output = self.img(x)

        return output


class TextDecoder(nn.Module):
    """
    Generate a piece of text given a hidden vector (encoding of source, either image or text)
    """
    def __init__(self, embed_dim, hidden_dim,
                 vocab_size, num_layers=1, decoder_model_name='lstm'):
        """Set the hyper-parameters and build the layers."""
        super(TextDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_dim)
        if decoder_model_name.lower() == 'lstm':
            self.model = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise NotImplementedError('Only LSTM is supported for TextDecoder')

        self.readout_layer = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.readout_layer.weight.data.uniform_(-0.1, 0.1)
        self.readout_layer.bias.data.fill_(0)

    def initialize_state(self, source_encoding, noise_z):
        batch_size = source_encoding.size(0)
        # h0 and c0 are of shape `(batch, num_layers * num_directions, hidden_size)
        c0 = Variable(torch.zeros(
            size=(1, batch_size, self.hidden_dim)
        ), requires_grad=False)
        if is_cuda:
            c0 = c0.cuda()
        # h0 is initialized by source encoding
        h0 = source_encoding.unsqueeze(0)

        # add a random noise to h0
        h0 = h0 + noise_z
        '''
        noise_type = 'normal'
        if noise_type == 'normal':
            # a normal noise
            noise_z = Variable(torch.randn(
                size=(1, batch_size, self.hidden_dim)
            ), requires_grad=False)
        else:
            # add a uniform noise of [min_encoding, max_encoding)
            max_embedding = torch.max(source_encoding).cpu() if is_cuda else torch.max(source_encoding)
            min_embedding = torch.min(source_encoding).cpu() if is_cuda else torch.min(source_encoding)
            # [batch_size, 1, hidden_dim]
            noise_z = (max_embedding - min_embedding) * torch.rand(size=(1, batch_size, self.hidden_dim)) \
                      + torch.FloatTensor([float(min_embedding)]).unsqueeze(0).unsqueeze(1)
        if is_cuda:
            noise_z = noise_z.cuda()
        h0 = h0 + noise_z
        '''

        return h0, c0


    def forward(self, image_encoding, text_words, text_lengths, noise_z):
        """
        Decode image feature vectors and generates captions.
        :param image_encoding: output of encoder, shape=[batch_size, hidden_dim]
        :param text_words: ground-truth of output text, for teacher forcing, shape=[batch_size, max_len]
        :param text_lengths:  length of each caption sequence in the batch
        :param initialize_noise:    whether to add noise (z) into the initial state of decoder
        :return:
        # return: outputs (s, V), lengths list(Tmax)
        """
        # truncate the last word and convert to embedding[batch_size, max_length-1, embed_dim]
        embed_words = self.embed(text_words[:,:-1])
        (h_0, c_0) = self.initialize_state(image_encoding, noise_z)

        self.model.flatten_parameters()
        # [batch_size, max_len-1, hidden_dim], ([1, batch_size, hidden_dim], [1, batch_size, hidden_dim])
        hidden_states, final_state = self.model(embed_words, (h_0, c_0))
        # [batch_size, max_len-1, vocab_size]
        word_logits = self.readout_layer(hidden_states)

        return word_logits, text_lengths


    def sample(self, image_encoding, noise_z, init_word, max_len):
        """
        Samples captions for given image features (Greedy search).
        :param image_encoding: [batch_size, hidden_dim]
        :param init_word: [batch_size, 1]
        :param max_len: an integer indicating the max length of predicted sequence
        :return:
        """
        batch_size = init_word.size(0)
        states = self.initialize_state(image_encoding, noise_z)
        sampled_words = []
        input_word = init_word
        for i in range(max_len):
            # [batch_size, 1, embed_dim]
            embed_words = self.embed(input_word)
            # hidden=[batch_size, 1, hidden_dim], states=([1, batch_size, hidden_dim], [1, batch_size, hidden_dim])
            hiddens, states = self.model(embed_words, states)
            # [batch_size, vocab_size]
            logits = self.readout_layer(hiddens.squeeze(1))
            # apply gumbel_softmax, predicted_word=[batch_size, vocab_size]
            probs = gumbel_softmax(logits, tau=0.96, hard=True)
            next_word = probs.max(dim=1)[1]
            # outputs = self.softmax(outputs)
            # predicted_index = outputs.multinomial(1)
            # predicted = outputs[predicted_index]
            sampled_words.append(next_word)
            input_word = next_word.unsqueeze(1)

        #sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
        # (batch_size, max_len)
        sampled_words = torch.stack(sampled_words, 1)
        # sampled_words = sampled_words.permute(1, 0)

        return sampled_words


    def pre_compute(self, features, gen_samples, eval_t, states=None):

        best_sample_nums = 5 # number of vocabs to sample

        inputs = features.unsqueeze(1) # (b, 1, e)
        if torch.cuda.is_available():
            gen_samples = gen_samples.type(torch.cuda.LongTensor)
        else:
            gen_samples = gen_samples.type(torch.LongTensor)
        forced_inputs = gen_samples[:,:eval_t]

        for i in range(eval_t):
            hiddens, states = self.model(inputs, states)          # hiddens = (b, 1, h)
            inputs = self.embed(forced_inputs[:,i])
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)

        outputs = self.readout_layer(hiddens.squeeze(1))
        outputs = self.softmax(outputs)
        predicted_indices = outputs.multinomial(best_sample_nums)

        return predicted_indices, states


    def rollout(self, features, gen_samples, t, Tmax, states=None):
        """
            sample caption from a specific time t

            features = (b, e)
            t = scalar
            Tmax = scalar
            states = cell states = tuple
        """
        sampled_ids = []
        # inputs = features.unsqueeze(1) # (b, 1, e)
        if torch.cuda.is_available():
            gen_samples = gen_samples.type(torch.cuda.LongTensor)
        else:
            gen_samples = gen_samples.type(torch.LongTensor)
        # forced_inputs = gen_samples[:,:t+1]
        # for i in range(t):
        #     hiddens, states = self.lstm(inputs, states)          # hiddens = (b, 1, h)
        #     inputs = self.embed(forced_inputs[:,i])
        #     inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)

        inputs = self.embed(gen_samples[:,t]).unsqueeze(1)
        for i in range(t, Tmax):                                 # maximum sampling length
            # pdb.set_trace()
            hiddens, states = self.model(inputs, states)          # hiddens = (b, 1, h)
            outputs = self.readout_layer(hiddens.squeeze(1))            # outputs = (b, V)
            predicted = outputs.max(1)[1]

            # pdb.set_trace()
            # TODO maybe need to sample?
            # outputs = self.softmax(outputs)
            # predicted_index = outputs.multinomial(1)
            # predicted = outputs[predicted_index]

            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)

        sampled_ids = torch.cat(sampled_ids, 0)                  # (batch_size, 20)
        # pdb.set_trace()
        sampled_ids = sampled_ids.view(-1, Tmax-t)
        return sampled_ids


class TextDiscriminator(nn.Module):
    """
    Discriminate if a text is genuine or not
    """
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers=1):
        super(TextDiscriminator, self).__init__()
        self.text_feature_encoder = TextEncoder(output_dim=hidden_dim, embed_dim=embed_dim,
                                                hidden_dim=hidden_dim, vocab_size=vocab_size,
                                                num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.output_layer.weight.data.uniform_(-0.1, 0.1)
        self.output_layer.bias.data.fill_(0)

    def forward(self, captions, lengths):
        text_features = self.text_feature_encoder(captions, lengths)

        logit = self.output_layer(text_features).squeeze(1)
        output_prob = nn.Sigmoid()(logit)

        return output_prob


class ImageDiscriminator(nn.Module):
    '''
    Each input is (nc) x image_size x image_size
    '''
    def __init__(self, image_size=64, encoding_dim=256):
        super(ImageDiscriminator, self).__init__()
        self.image_size = image_size
        self.encoding_dim = encoding_dim
        self.num_channels = 3

        self.image_encoder = ImageEncoder(image_size=self.image_size, output_dim=self.encoding_dim)
        self.output_layer = nn.Linear(self.encoding_dim, 1)


    def forward(self, image):
        # [batch_size, encoding_dim]
        x_encoding = self.image_encoder(image)

        logit = self.output_layer(x_encoding).squeeze(1)
        real_prob = nn.Sigmoid()(logit)

        return real_prob

"""
class ImageDiscriminator128(nn.Module):
    '''
    input is (nc) x 128 x 128
    '''

    def __init__(self, input_dim):
        super(ImageDiscriminator128, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.input_dim = input_dim
        self.projected_embed_dim = 128
        self.ndf = 64
        self.B_dim = 128
        self.C_dim = 16

        self.image_encoder = ImageEncoder(image_size=128, output_dim=128)
        self.projector = Concat_embed(self.input_dim, self.projected_embed_dim)

        self.outlogitscond = nn.Sequential(
            conv3x3(self.ndf * 8 + self.projected_embed_dim, self.ndf * 8),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid()
        )

        # self.outlogits = nn.Sequential(
        #     nn.Conv2d(self.ndf * 8, 1, kernel_size=4, stride=4),
        #     nn.Sigmoid()
        # )



    def forward(self, inp, embed):
        '''
        :param inp: an image of [bs, nc=3, 128, 128]
        :param embed: an encoding in shape of [bs, self.embed_dim]
        :return:
        '''
        x_intermediate = self.image_encoder(inp)
        x = self.projector(x_intermediate, embed)
        x_cond = self.outlogitscond(x)
        # x_uncond = self.outlogits(x_intermediate)

        return x_cond.view(-1, 1).squeeze(1)
"""


class ImageTextPairDiscriminator(nn.Module):
    """
    Not used now
    Discriminate if a pair of image and text is genuine or not
    """
    def __init__(self, image_size, embed_dim, hidden_dim, vocab_size, num_layers=1):
        super(ImageTextPairDiscriminator, self).__init__()
        self.image_feature_encoder = ImageEncoder(image_size=image_size, output_dim=hidden_dim)
        self.text_feature_encoder = TextEncoder(output_dim=hidden_dim, embed_dim=embed_dim,
                                                hidden_dim=hidden_dim, vocab_size=vocab_size,
                                                num_layers=num_layers)


    def forward(self, images, captions, lengths):
        """
        Calculate reward score: r = logistic(dot_prod(f, h))
        """
        image_features = self.image_feature_encoder(images)
        text_features = self.text_feature_encoder(captions, lengths)

        # [batch_size, 1, 1] -> [batch_size]
        dot_prod = torch.bmm(image_features.unsqueeze(1), text_features.unsqueeze(1).transpose(2,1)).squeeze()
        output_prod = nn.Sigmoid()(dot_prod)

        return output_prod

    def forward_interpolate(self, images, caption_embeds):
        """
        Feeding embeddings of text directly
        Calculate reward score: r = logistic(dot_prod(f, h))
        """
        image_features = self.image_feature_encoder(images)
        text_features = caption_embeds

        dot_prod = torch.bmm(image_features.unsqueeze(1), text_features.unsqueeze(1).transpose(2,1)).squeeze()
        output_prod = nn.Sigmoid()(dot_prod)

        return output_prod


class TextEncoder(nn.Module):
    def __init__(self, output_dim, embed_dim,
                 hidden_dim, vocab_size,
                 dropout=0.0,
                 num_layers=1,
                 model_name='lstm'):
        """Set the hyper-parameters and build the layers."""
        super(TextEncoder, self).__init__()
        self.model_name = model_name.lower()

        print('vocab=%d, embed_dim=%d, hidden_dim=%d, output_dim=%d' % (vocab_size, embed_dim, hidden_dim, output_dim))

        if self.model_name == 'lstm':
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.model = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
            self.output_size = hidden_dim
        # elif self.model_name == 'skipthought':
        #     assert vocab, "vocab must be provided to load model"
        #     assert skip_thoughts_path, "skip_thoughts_path must be set to load model"
        #     self.model = UniSkip(skip_thoughts_path, vocab, dropout=dropout)
        #     self.output_size = 2400
        else:
            raise NotImplementedError('Only LSTM is supported for TextDecoder')

        self.output_bridge = nn.Linear(hidden_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.output_bridge.weight.data.uniform_(-0.1, 0.1)
        self.output_bridge.bias.data.fill_(0)

    def forward(self, text_words, text_lengths):
        """
        Decode image feature vectors and generates captions.
        :param encoder_features: output of encoder, shape=[batch_size, hidden_dim]
        :param text_words: ground-truth of output text, for teacher forcing, shape=[batch_size, max_len]
        :param text_lengths:  length of each caption sequence in the batch
        :param initialize_noise:    whether to add noise (z) into the initial state of decoder
        :return:
        """
        # [batch_size, max_length, embed_dim]

        if self.model_name == 'lstm':
            text_embeds = self.embed(text_words)
            packed_embeds = pack_padded_sequence(text_embeds, text_lengths, batch_first=True)
            # final_states = (h_n, c_n)
            memory_bank, final_states = self.model(packed_embeds)
            # use the h_n as the final encoding
            final_encoding = final_states[0].squeeze(0)
            # final_encoding = [pad_packed_sequence(memory_bank, batch_first=True)[0][i][len_i - 1].data.numpy() for i, len_i in enumerate(pad_packed_sequence(memory_bank, batch_first=True)[1])]
        # elif self.model_name == 'skipthought':
        #     final_encoding = self.model(input, lengths=text_lengths).data.numpy()

        # [batch_size, output_dim]
        output_encoding = self.output_bridge(final_encoding)

        return output_encoding


class ImageEncoder(nn.Module):
    '''
    Feeding an image of 64*64, return a vector of dim=hidden_dim
    Architecture is based on the discriminator of DCGAN
    '''

    def __init__(self, image_size=64, output_dim=128):
        super(ImageEncoder, self).__init__()
        self.image_size = image_size
        self.num_channels = 3
        self.output_dim = output_dim
        # self.ndf = 64
        self.ndf = 16
        # self.B_dim = 128
        # self.C_dim = 16

        self.vision_feature_dim = 128 * self.ndf
        if self.image_size == 64:
            self.model = nn.Sequential(
                # * input is (nc) x 64 x 64
                nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                # * state size. (ndf) x 32 x 32
                nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=True),
                nn.BatchNorm2d(self.ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # * state size. (ndf*2) x 16 x 16
                nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=True),
                nn.BatchNorm2d(self.ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # * state size. (ndf*4) x 8 x 8
                nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=True),
                nn.BatchNorm2d(self.ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                # below layers are different from DCGAN
                nn.Conv2d(self.ndf * 8, self.ndf * 2, 1, 1, 0),
                nn.BatchNorm2d(self.ndf * 2),
                nn.LeakyReLU(0.2, True),

                nn.Conv2d(self.ndf * 2, self.ndf * 2, 3, 1, 1),
                nn.BatchNorm2d(self.ndf * 2),
                nn.LeakyReLU(0.2, True),

                nn.Conv2d(self.ndf * 2, self.ndf * 8, 3, 1, 1),
                nn.BatchNorm2d(self.ndf * 8),
                nn.LeakyReLU(0.2, True)

                # output size (ndf*8) x 8 x 8
            )
        elif self.image_size == 128:
            self.model = nn.Sequential(
                # state size = 3 x 128 x 128
                nn.Conv2d(3, self.ndf, 4, 2, 1, bias=True),  # 128 * 128 * ndf
                nn.LeakyReLU(0.2, inplace=True),
                # state size = ndf x 64 x 64
                nn.Conv2d(self.ndf, self.ndf * 2, 3, 1, 1, bias=True),
                nn.BatchNorm2d(self.ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=True),
                nn.BatchNorm2d(self.ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),  # 32 * 32 * ndf * 4
                nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=True),
                nn.BatchNorm2d(self.ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),  # 16 * 16 * ndf * 8
                nn.Conv2d(self.ndf * 8, self.ndf * 8, 3, 1, 1),
                nn.BatchNorm2d(self.ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),  # 8 * 8 * ndf * 8
                nn.Conv2d(self.ndf * 8, self.ndf * 2, 4, 2, 1),
                nn.BatchNorm2d(self.ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),  # 4 * 4 * ndf * 2
                nn.Conv2d(self.ndf * 2, self.ndf * 8, 3, 1, 1),
                nn.BatchNorm2d(self.ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.ndf * 8, self.ndf * 8, 4, 2, 1),
                nn.BatchNorm2d(self.ndf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                # nn.Conv2d(self.ndf * 8, self.ndf * 16, 4, 2, 1, bias=True),
                # nn.BatchNorm2d(self.ndf * 16),
                # nn.LeakyReLU(0.2, inplace=True),  # 8 * 8 * ndf * 16
                # nn.Conv2d(self.ndf * 16, self.ndf * 32, 4, 2, 1, bias=True),
                # nn.BatchNorm2d(self.ndf * 32),
                # nn.LeakyReLU(0.2, inplace=True),  # 4 * 4 * ndf * 32
                # conv3x3(self.ndf * 32, self.ndf * 16),
                # nn.BatchNorm2d(self.ndf * 16),
                # nn.LeakyReLU(0.2, inplace=True),   # 4 * 4 * ndf * 16
                # nn.Conv2d(self.ndf * 16, self.ndf * 8, 3, 1, 1, bias=True),
                # nn.BatchNorm2d(self.ndf * 8),
                # nn.LeakyReLU(0.2, inplace=True)   # 4 * 4 * ndf * 8
            )
        else:
            raise NotImplementedError('image_size of ImageEncoder can only be 64 or 128')

        self.bridge_layer = nn.Linear(self.vision_feature_dim, self.output_dim)
        self.bn = nn.BatchNorm1d(self.output_dim, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.bridge_layer.weight.data.normal_(0.0, 0.02)
        self.bridge_layer.bias.data.fill_(0)

    def forward(self, image_inp):
        '''
        :param inp: an image of [bs, nc=3, 128, 128]
        :param embed: an encoding in shape of [bs, self.embed_dim]
        :return:
        '''
        image_encoding = self.model(image_inp)
        image_encoding = image_encoding.view(image_encoding.size(0), -1)
        image_encoding = self.bn(self.bridge_layer(image_encoding))

        return image_encoding
