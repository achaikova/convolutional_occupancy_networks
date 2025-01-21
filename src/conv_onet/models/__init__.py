import torch
import torch.nn as nn
from torch import distributions as dist
from src.conv_onet.models import decoder

# Decoder dictionary
decoder_dict = {
    'simple_local': decoder.LocalDecoder,
    'simple_local_crop': decoder.PatchLocalDecoder,
    'simple_local_point': decoder.LocalPointDecoder
}


class ConvolutionalOccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
        embedding_mode (str): where to use embeddings ('encoder', 'decoder', 'none')
        num_classes (int): number of classes for label embedding
        embedding_dim (int): dimension of label embeddings
    '''

    def __init__(self, decoder, encoder=None, device=None, 
                 embedding_mode='none', num_classes=0, embedding_dim=0):
        super().__init__()
        
        self.decoder = decoder.to(device)
        self.embedding_mode = embedding_mode

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        # Initialize label embedding if needed
        if embedding_mode != 'none' and num_classes > 0 and embedding_dim > 0:
            self.label_embedding = LabelEmbedding(num_classes, embedding_dim).to(device)
        else:
            self.label_embedding = None

        self._device = device

    def forward(self, p, inputs, sample=True, labels=None, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
            labels (tensor): class labels for embedding
        '''
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)

        # Get embeddings if label embedding is enabled
        embeddings = None
        if self.label_embedding is not None and labels is not None:
            embeddings = self.label_embedding(labels)

        c = self.encode_inputs(inputs)

        p_r = self.decode(p, c, embeddings=embeddings, embedding_mode=self.embedding_mode, **kwargs)
        return p_r

    def encode_inputs(self, inputs, embeddings=None, embedding_mode='none'):
        ''' Encodes the input.

        Args:
            inputs (tensor): the input
            embeddings (tensor): text embeddings if available
            embedding_mode (str): how to handle embeddings ('encoder_cat', 'encoder_add', 'none')
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def decode(self, p, c, embeddings=None, embedding_mode='none', **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, c, embeddings=embeddings, embedding_mode=embedding_mode, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model


class LabelEmbedding(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
    
    def forward(self, labels):
        return self.embedding(labels)
