import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import *
from torch.autograd import Variable
import pdb
from tqdm import tqdm

is_cuda = torch.cuda.is_available()

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        # resnet = models.resnet152(pretrained=True)
        # modules = list(resnet.children())[:-1]      # delete the last fc layer.
        
        # self.resnet = nn.Sequential(*modules)
        # self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        
        alexnet = models.alexnet(pretrained=True)
        self.alexnet_features = alexnet.features
        # modules = list(alexnet.classifier.children())[:-1]      # delete the last fc layer.

        # self.alexnet_classifier = nn.Sequential(*modules)
        # self.linear = nn.Linear(list(alexnet.children())[-1][-1].in_features, embed_size)
        self.linear = nn.Linear(2304, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        
    def forward(self, images):
        """Extract the image feature vectors."""
        #features = self.resnet(images)
        features = self.alexnet_features(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        # features = self.alexnet_classifier(features)
        # features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.readout_layer = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.readout_layer.weight.data.uniform_(-0.1, 0.1)
        self.readout_layer.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths, initialize_noise=False):
        """
        Decode image feature vectors and generates captions.
        :param features: output of encoder, shape=[batch_size, hidden_dim]
        :param captions: ground-truth of output text, for teacher forcing, shape=[batch_size, max_len]
        :param lengths:  length of each caption sequence in the batch
        :param initialize_noise:    whether to add noise (z) into the initial state of decoder
        :return:
        # return: outputs (s, V), lengths list(Tmax)
        """
        # [batch_size, max_length, embed_dim]
        embeddings = self.embed(captions)
        # concatenate image encoding to the beginning of embeddings as the 1st input state to decoder
        if not initialize_noise:
            # [batch_size, max_length+1, embed_dim]
            embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        else:
            # add a uniform noise of [min_encoding, max_encoding) to
            max_embedding = torch.max(features).cpu() if is_cuda else torch.max(features)
            min_embedding = torch.min(features).cpu() if is_cuda else torch.min(features)
            # [batch_size, 1, hidden_dim]
            noise_z = (max_embedding - min_embedding) * torch.rand(size=(embeddings.size(0), 1, embeddings.size(2))) + torch.FloatTensor([float(min_embedding)]).unsqueeze(0).unsqueeze(1)
            noise_z = noise_z.cuda() if is_cuda else noise_z
            # [batch_size, 2, hidden_dim]
            features = torch.cat((features.unsqueeze(1), Variable(noise_z, requires_grad=False)), dim=1)
            # [batch_size, max_length+2, embed_dim]...
            embeddings = torch.cat((features, embeddings), 1)
        # seems not necessary to pack in decoding
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        packed_hiddens, hidden_final = self.lstm(packed)
        # [batch_size, max_len, hidden_dim]
        hiddens, packed_lengths = pad_packed_sequence(packed_hiddens, batch_first=True)
        # [batch_size, max_len, vocab_size]
        logits = self.readout_layer(hiddens)
        return logits, packed_lengths
    
    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.readout_layer(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            # pdb.set_trace()
            # outputs = self.softmax(outputs)
            # predicted_index = outputs.multinomial(1)
            # predicted = outputs[predicted_index]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        #sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
        sampled_ids = torch.cat(sampled_ids, 0)                  # (batch_size, 20)
        sampled_ids = sampled_ids.view(-1, 20)
        # return sampled_ids.squeeze()
        # pdb.set_trace()
        return sampled_ids


    def pre_compute(self, features, gen_samples, eval_t, states=None):
        
        best_sample_nums = 5 # number of vocabs to sample

        inputs = features.unsqueeze(1) # (b, 1, e)
        if torch.cuda.is_available():
            gen_samples = gen_samples.type(torch.cuda.LongTensor)
        else:
            gen_samples = gen_samples.type(torch.LongTensor)
        forced_inputs = gen_samples[:,:eval_t]

        for i in range(eval_t):
            hiddens, states = self.lstm(inputs, states)          # hiddens = (b, 1, h)
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
            hiddens, states = self.lstm(inputs, states)          # hiddens = (b, 1, h)
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
