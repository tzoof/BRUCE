import torch
import os
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data.dataset import Dataset


class BundlesTrainDataSet(Dataset):
    def __init__(self, dataset_obj, dataset_size):
        super().__init__()
        self.dataset_size = dataset_size
        self.users, self.positives, self.negatives = get_train_bundles(dataset_obj, dataset_size)

    def __getitem__(self, index):
        return self.users[index], self.positives[index], self.negatives[index]

    def __len__(self):
        return self.dataset_size


class ItemsTrainDataSet(Dataset):
    def __init__(self, dataset_obj, dataset_size):
        super().__init__()
        self.dataset_size = dataset_size
        self.users, self.positives, self.negatives = get_train_items(dataset_obj, dataset_size)

    def __getitem__(self, index):
        return self.users[index], self.positives[index], self.negatives[index]

    def __len__(self):
        return self.dataset_size


class BundlesRepresentationDataset(Dataset):
    def __init__(self, bundles_representation, attention_masks):
        super().__init__()
        self.bundles_representation = bundles_representation
        self.attention_masks = attention_masks

    def __getitem__(self, index):
        return index, self.bundles_representation[index], self.attention_masks[index]

    def __len__(self):
        return self.bundles_representation.shape[0]


class ItemItemDatasetForPretrain(Dataset):
    def __init__(self, dataset_name, set):
        super().__init__()
        self.items, self.labels = create_pretrain_dataset(dataset_name, set)

    def __getitem__(self, index):
        return self.items[index], self.labels[index]

    def __len__(self):
        return self.labels.shape[0]


def get_train_bundles(dataset_obj, dataset_size):
    """ return 3 lists of size <size> - user, pos, neg.
    each item in this lists is a user id or an bundle id.
    for index 0<j<size:  user[j] bought bundle pos[j] and didn't buy bundle neg[j].
    """
    train_bundle_u, train_bundle_b = dataset_obj.ground_truth_u_b_train.nonzero()
    positive_interactions_indices = np.random.randint(0, high=len(train_bundle_u), size=dataset_size)
    users = train_bundle_u[positive_interactions_indices]
    pos = train_bundle_b[positive_interactions_indices]
    # The negative items are chosen randomly from the items that weren't bought.
    neg = np.random.choice(range(dataset_obj.num_bundles), dataset_size, replace=True)

    # batch_size is also the len of users and pos
    for i in range(dataset_size):
        # item is the list of items ids that were bought by the user
        user_positive_bundles_train = dataset_obj.ground_truth_u_b_train[users[i]].nonzero()[1]
        while neg[i] in user_positive_bundles_train:
            neg[i] = np.random.choice(range(dataset_obj.num_bundles))
    return torch.from_numpy(users), torch.from_numpy(pos), torch.from_numpy(neg)


def get_train_items(dataset_obj, dataset_size):
    """ return 3 lists of size <batch_size> - user, pos, neg.
    each item in this lists is a user id or an item id (int).
    for index 0<j<size:  user[j] bought item pos[j] and didn't buy item neg[j].
    """
    train_item_u, train_item_i = dataset_obj.ground_truth_u_i.nonzero()
    positive_interactions_indices = np.random.randint(0, high=len(train_item_u), size=dataset_size)
    users = train_item_u[positive_interactions_indices]
    pos = train_item_i[positive_interactions_indices]
    # The negative items are chosen randomly from the items that weren't bought.
    neg = np.random.choice(range(dataset_obj.num_items), dataset_size, replace=True)
    for i in range(dataset_size):
        # item is the list of items ids that were bought by the user
        user_positive_items = dataset_obj.ground_truth_u_i[users[i]].nonzero()[1]
        while neg[i] in user_positive_items:
            neg[i] = np.random.choice(range(dataset_obj.num_items))
    return torch.from_numpy(users), torch.from_numpy(pos), torch.from_numpy(neg)


def get_bundles_representation(dataset, args, item_padding_id, cls_token, device):
    # Bert without bundleEmbedding : we add a cls token in the beginning of the bundle and pass it as the output of the transformer to the dense layer
    # Bert with bundleEmbedding : in PostUL - the same, since the bundle embedding is added only after the transformers
    # In PreUL - we don't add the cls since the bundle embedding is used as the cls and it is the one that is passed to the dense layer
    bundles_representation = []
    attention_masks = []
    for b in range(dataset.num_bundles):
        # Take only the first max_bundle_size items in the bundle
        items_in_bundle = dataset.ground_truth_b_i[b].nonzero()[1][:args.max_bundle_size]
        bundle_size = len(items_in_bundle)
        # In both bert and PreUL+bundle we add a non empty item
        if args.op_after_transformer == 'bert' or (args.bundleEmbeddings and not args.usePostUL):
            attention_mask = [1]*(bundle_size+1) + [0]*(args.max_bundle_size - bundle_size)
        else:
            attention_mask = [1]*(bundle_size) + [0]*(args.max_bundle_size - bundle_size)
        items_in_bundle = torch.tensor(items_in_bundle, dtype=torch.long)
        items_in_bundle = nn.functional.pad(items_in_bundle, (0, args.max_bundle_size - bundle_size),
                                            mode='constant', value=item_padding_id)
        # If we use bundle embedding with PreUL - we don't need a cls token - we use the embedding as cls token
        if args.op_after_transformer == 'bert' and not (args.bundleEmbeddings and not args.usePostUL):
            items_in_bundle = nn.functional.pad(items_in_bundle, (1, 0),
                                                mode='constant', value=cls_token)
        bundles_representation.append(items_in_bundle)
        attention_masks.append(torch.tensor(attention_mask, dtype=torch.long, device=device))
    return torch.stack(bundles_representation).to(device=device), torch.stack(attention_masks).unsqueeze(2)


def df_to_tensor(df, device):
    return torch.from_numpy(df.values).float().to(device)


def create_pretrain_dataset(dataset_name, set):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert(set in ["train", "val", "test"])
    path = os.path.join("Data", dataset_name, f"item_item_{set}.txt")
    df = pd.read_csv(path, delimiter="\t", names=["item1", "item2", 'label'])
    inputs = torch.from_numpy(df[["item1", "item2"]].values).long().to(device)
    labels = torch.from_numpy(df["label"].values).float().to(device)
    return inputs, labels