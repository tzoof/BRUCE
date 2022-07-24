import torch
from torch import nn
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder


""" Model Architecture"""
class Model(nn.Module):
    def __init__(self, args, num_users, num_items, num_bundles):
        super(Model, self).__init__()
        self.args = args
        self.max_bundle_size = args.max_bundle_size
        self.embed_shape = args.embed_shape

        # The embedding is of size #items + 1 to represent padding
        self.item_embeddings = nn.Embedding(num_items + 1, self.embed_shape)
        self.user_embeddings = nn.Embedding(num_users + 2, self.embed_shape)

        self.items_embedding_pipe = nn.Sequential(self.item_embeddings)
        self.user_embedding_pipe = nn.Sequential(self.user_embeddings)

        self.transformer_layer = TransformerEncoderLayer(d_model=args.embed_shape, nhead=3,
                                                         dim_feedforward=args.transformer_feedforward_dim)

        self.bundle_representation = nn.Sequential(nn.Dropout(p=args.transformer_dropout), Transpose(),
                                                   TransformerEncoder(self.transformer_layer, args.num_transformer_layers),
                                                   Transpose(), nn.Dropout(p=args.transformer_dropout))

        self.item_dense1 = nn.Linear(args.embed_shape * 2, args.embed_shape)
        self.item_prediction = nn.Linear(args.embed_shape, 1)
        self.bundle_prediction = nn.Linear(args.embed_shape, 1)
        self.item_item_prediction = nn.Linear(args.embed_shape, 1)
        self.init_weights()

    def init_weights(self):
        initrange = self.args.weights_initrange
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)

        for layer in [self.item_prediction, self.bundle_prediction, self.item_item_prediction]:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()


    def forward(self, x, user, bundle_ids=None, bundle=False, attention_masks=None, pretrain=False):
        ''' In training: the forward method rceives a tensor of shape (batch_size, max_bundle_size)
        The user shape is (batch_size, 1), 1 use id for each user
        The bundle_ids shape is (batch_size, 1), 1 use id for each user
        In eval: the forward method rceives a tensor of shape (1, max_bundle_size)
        The user shape is (#users, 1), 1 use id for each user, #users can very between 1 to 99
        For items prediction only the first item in dim 1 is full.
        For bundles with i items, there will be i non empty objects in dim 1'''
        current_batch_size = x.shape[0]
        assert x.shape == (current_batch_size, self.max_bundle_size) or x.shape == (current_batch_size, 2)
        current_items_embeddings = self.items_embedding_pipe(x)
        current_user_embeddings = self.user_embedding_pipe(user)
        x = torch.clone(current_items_embeddings)
        user_emb = torch.clone(current_user_embeddings)
        if len(user_emb.shape) == 2:
            user_emb = user_emb.unsqueeze(dim=1)
        # Adding the user embedding as a first item
        if attention_masks is not None:
            x *= attention_masks


        # transformer
        if bundle:
            x = torch.cat(tensors=(user_emb, x), dim=1)
            x = self.bundle_representation(x)
            x = x[:, 0, :]
        else:
            x = x[:, 0, :].unsqueeze(dim=1)
            x = torch.cat(tensors=(user_emb, x), dim=2)
            x = torch.relu(self.item_dense1(x))

        # Bert like aggregation
        if bundle:
            x = torch.relu(self.bundle_prediction(x))
        else:
            x = torch.relu(self.item_prediction(x))
        if self.training:
            users_regularization = current_user_embeddings.square().sum(-1).mean()
            items_regularization = current_items_embeddings.square().sum(-1).mean()
            return x, users_regularization, items_regularization
        else:
            return x

class Transpose(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.transpose(1, 0)
        return x

