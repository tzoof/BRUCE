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
        if self.args.op_after_transformer == 'bert':
            self.item_embeddings = nn.Embedding(num_items + 2, self.embed_shape)
        else:
            self.item_embeddings = nn.Embedding(num_items + 1, self.embed_shape)
        self.user_embeddings = nn.Embedding(num_users + 1, self.embed_shape)
        self.bundle_embeddings = nn.Embedding(num_bundles + 1, self.embed_shape)

        self.items_embedding_pipe = nn.Sequential(self.item_embeddings)
        self.user_embedding_pipe = nn.Sequential(self.user_embeddings)
        self.bundle_embedding_pipe = nn.Sequential(self.bundle_embeddings)

        self.transformer_layer = TransformerEncoderLayer(d_model=args.embed_shape, nhead=3,
                                                         dim_feedforward=args.transformer_feedforward_dim)

        self.bundle_representation = nn.Sequential(nn.Dropout(p=args.transformer_dropout), Transpose(),
                                                   TransformerEncoder(self.transformer_layer, args.num_transformer_layers),
                                                   Transpose(), nn.Dropout(p=args.transformer_dropout))

        self.item_prediction = nn.Linear(args.embed_shape * 2, 1)
        # +1 for the user emb +1 for the bundle embedding
        if self.args.op_after_transformer == 'concat':
            self.bundle_prediction = nn.Linear((args.max_bundle_size + 2) * self.embed_shape, 1)
        elif self.args.op_after_transformer == 'avg' or self.args.op_after_transformer == 'bert':
            self.bundle_prediction = nn.Linear(args.embed_shape * 3, 1)
        else:
            raise Exception("op_after_transformer argument must be within ['concat', 'avg', 'bert']")
        self.init_weights()

    def init_weights(self):
        initrange = self.args.weights_initrange
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.bundle_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_prediction.weight.data.uniform_(-initrange, initrange)
        self.item_prediction.bias.data.zero_()
        self.bundle_prediction.weight.data.uniform_(-initrange, initrange)
        self.bundle_prediction.bias.data.zero_()

    def forward(self, x, user, bundle_ids=None, bundle=False, attention_masks=None):
        ''' In training: the forward method rceives a tensor of shape (batch_size, max_bundle_size)
        The user shape is (batch_size, 1), 1 use id for each user
        The bundle_ids shape is (batch_size, 1), 1 use id for each user
        In eval: the forward method rceives a tensor of shape (1, max_bundle_size)
        The user shape is (#users, 1), 1 use id for each user, #users can very between 1 to 99
        For items prediction only the first item in dim 1 is full.
        For bundles with i items, there will be i non empty objects in dim 1'''
        current_batch_size = x.shape[0]
        assert (x.shape == (current_batch_size, self.max_bundle_size) or
                (self.args.op_after_transformer == "bert" and x.shape == (current_batch_size, self.max_bundle_size + 1)))
        current_items_embeddings = self.items_embedding_pipe(x)
        current_user_embeddings = self.user_embedding_pipe(user)
        x = torch.clone(current_items_embeddings)

        if bundle:
            if bundle_ids is None:
                raise Exception("The bundle_id wasn't passed to the model")
            bundle_emb = self.bundle_embedding_pipe(bundle_ids)
            if attention_masks is not None:
                x *= attention_masks
            x = self.bundle_representation(x)
            if self.args.op_after_transformer == 'avg':
                x = x.mean(dim=1)
                x = torch.cat(tensors=(x, current_user_embeddings.squeeze(dim=1), bundle_emb.squeeze(dim=1)), dim=1)
            elif self.args.op_after_transformer == 'sum':
                x = x.sum(dim=1)
                x = torch.cat(tensors=(x, current_user_embeddings.squeeze(dim=1), bundle_emb.squeeze(dim=1)), dim=1)
            elif self.args.op_after_transformer == 'concat':
                x = x.flatten(1)
                x = torch.cat(tensors=(x, current_user_embeddings.squeeze(dim=1), bundle_emb.squeeze(dim=1)), dim=1)
            elif self.args.op_after_transformer == 'bert':
                x = x[:, 0, :]
                x = torch.cat(tensors=(x, current_user_embeddings.squeeze(dim=1), bundle_emb.squeeze(dim=1)), dim=1)
            else:
                raise Exception("op_after_transformer argument must be within ['concat', 'avg', 'bert', 'sum']")
            x = torch.relu(self.bundle_prediction(x))
            if self.training:
                users_regularization = current_user_embeddings.square().sum(-1).mean()
                items_regularization = current_items_embeddings.square().sum(-1).mean()
                return x, users_regularization, items_regularization
            else:
                return x

        # Only the first item in dim 1 is non zero and represents the item embedding
        x = x[:, 0, :]
        x = torch.cat(tensors=(x, current_user_embeddings), dim=1)
        assert (x.shape == (current_batch_size, self.embed_shape * 2))
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

