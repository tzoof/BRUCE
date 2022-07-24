from typing import Any

import torch
from torch import nn


class Model(nn.Module):
    """ Model Architecture
    A simple classification model that receives 2 items ids and predict whether they ever co-occurred in a bundle.
    Consist on item embedding (will be shared with the ranking model) and a dense layer wth sigmoid.
    """
    def __init__(self, args, num_items):
        super(Model, self).__init__()
        self.args = args
        self.embed_shape = args.embed_shape
        # For BERT models - uncomment the next line and comment the one after it
        #self.item_embeddings = nn.Embedding(num_items + 2, self.embed_shape)
        self.item_embeddings = nn.Embedding(num_items + 1, self.embed_shape)
        self.item_prediction1 = nn.Linear(args.embed_shape * 2, args.embed_shape * 2)
        self.item_prediction2 = nn.Linear(args.embed_shape * 2, 1)

        self.init_weights()

    def init_weights(self):
        initrange = self.args.weights_initrange
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        for layer in [self.item_prediction1, self.item_prediction2]:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()

    def forward(self, x):
        """The forward method receives a tensor of shape (batch_size, 2)
        Each is an item
        We replace each item with its embeddings and predict whether the 2 items co-occur in a bundle.
        """
        current_batch_size = x.shape[0]
        x = self.item_embeddings(x).reshape(current_batch_size, self.embed_shape * 2)
        x = torch.relu(self.item_prediction1(x))
        x = torch.sigmoid(self.item_prediction2(x))
        return x
