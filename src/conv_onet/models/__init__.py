import torch
import torch.nn as nn
from torch import distributions as dist
from src.conv_onet.models import decoder
from sentence_transformers import SentenceTransformer

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
        embedding_model (str): embedding model name (SentenceTransformer)
        embedding_mode (str): where to use embeddings ('cat', 'add', 'none')
        num_classes (int): number of classes for label embedding
        embedding_dim (int): dimension of label embeddings
    '''

    def __init__(self, decoder, encoder=None, device=None, 
                 embedding_mode='none', num_classes=0, embedding_dim=0, embedding_model=None):
        super().__init__()
        
        self.decoder = decoder.to(device)
        self.embedding_mode = embedding_mode

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        # Initialize label embedding if needed
        self.embedding_model = embedding_model
        if embedding_model is not None:
            self.label_embedding = SentenceTransformer(embedding_model).to(device)
            self.reduce_embedding = nn.ModuleList([
                nn.Linear(384, 128).to(device),
                nn.Linear(128, 32).to(device)
            ]).to(device)
            
        else:
            if embedding_mode != 'none' and num_classes > 0 and embedding_dim > 0:
                self.label_embedding = LabelEmbedding(num_classes, embedding_dim).to(device)
            else:
                self.label_embedding = None

        self._device = device
        self.embedding_mode = embedding_mode

    def create_embeddings(self, labels):
        if self.embedding_model is not None:
            embeddings = self.label_embedding.encode(labels['category_name'])
            embeddings = embeddings.mean(dim=1)
            print("embeddings after mean", embeddings.shape)
            embeddings = self.reduce_embedding[0](embeddings)
            embeddings = self.reduce_embedding[1](embeddings)
        elif self.label_embedding is not None:
            embeddings = self.label_embedding(labels['category_id'])
        else:
            embeddings = None
        return embeddings

    def forward(self, p, inputs, sample=True, labels: dict = None, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
            labels (dict): class labels for embedding
        '''
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)

        c = self.encode_inputs(inputs)

        p_r = self.decode(p, c, labels, **kwargs)
        return p_r

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            inputs (tensor): the input
            embeddings (tensor): text embeddings if available
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def decode(self, p, c, labels: dict = None, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
            labels (dict): class labels for embedding
        '''
        embeddings = self.create_embeddings(labels)
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
