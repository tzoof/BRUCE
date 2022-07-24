import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
import os
import enum
import scipy.sparse as sp
import torch
import pickle


class Product(enum.Enum):
    item = 1
    bundle = 2


class InefficientBundleTestDataset:
    def __init__(self, baseDataset):
        self.baseDatabase = baseDataset

    def __getitem__(self, index):
        return torch.tensor([index]), torch.from_numpy(self.baseDatabase.ground_truth_u_b_all_test[index].toarray()).squeeze(),  \
            torch.from_numpy(self.baseDatabase.ground_truth_u_b_train[index].toarray()).squeeze(),  \


    def __len__(self):
        return self.baseDatabase.num_users


class BundleTestDataset:
    def __init__(self, baseDataset):
        self.baseDatabase = baseDataset

    def __getitem__(self, index):
        return self.baseDatabase.test_relevant_users[index], torch.from_numpy(self.baseDatabase.ground_truth_u_b_test_relevant_users[index].toarray()).squeeze(),  \
            torch.from_numpy(self.baseDatabase.train_mask_only_relevant_test[index].toarray()).squeeze(),  \


    def __len__(self):
        return self.baseDatabase.ground_truth_u_b_test_relevant_users.shape[0]


class Dataset:
    """
    The dataset class, supports all the datasets (according to the dataset_string argument).
    Contain user-bundle, user-item, and item-bundle related properties.
    The class also supports graph sampling if use_graph_sampling is passed, but it did not improve our results and so was not used.
    """
    def __init__(self, path, args, use_mini_test=False):
        use_graph_sampling_cache = True
        self.path = path
        self.dataset_name = args.dataset_string
        self.num_users, self.num_bundles, self.num_items  = self.__load_data_size()

        self.initiate_bundle_items_map()
        self.initiate_user_item_properties()
        self.initiate_train_user_bundle_properties(use_graph_sampling_cache)
        self.initiate_test_user_bundle_properties(use_mini_test)
        self.initiate_tune_user_bundle_properties()
        if args.use_graph_sampling:
            if use_graph_sampling_cache:
                self.items_train_triplets = self.load_data_triplets(product_type=Product.item)
                self.bundle_train_triplets = self.load_data_triplets(product_type=Product.bundle)
                self.num_user_item_triplets = len(self.items_train_triplets)
                self.num_user_bundle_triplets = len(self.bundle_train_triplets)
            else:
                self.items_train_triplets = self.create_data_triplets(product_type=Product.item)
                self.bundle_train_triplets = self.create_data_triplets(product_type=Product.bundle)
                self.num_user_item_triplets = len(self.items_train_triplets)
                self.num_user_bundle_triplets = len(self.bundle_train_triplets)
                self.triplets_to_csv(self.items_train_triplets, Product.item)
                self.triplets_to_csv(self.bundle_train_triplets, Product.bundle)

    def load_obj(self, name):
        with open(name, 'rb') as f:
            return pickle.load(f)

    def __load_data_size(self):
        with open(os.path.join(self.path, self.dataset_name, f'{self.dataset_name}_data_size.txt'), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:3]

    def load_U_B_interaction(self, task, use_mini=False):
        file_name = f"user_bundle_{task}-minimini.txt" if use_mini else f"user_bundle_{task}.txt"
        with open(os.path.join(self.path, self.dataset_name, file_name), 'r') as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

    def load_U_I_interaction(self):
        with open(os.path.join(self.path, self.dataset_name, 'user_item.txt'), 'r') as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

    def load_B_I_interaction(self):
        with open(os.path.join(self.path, self.dataset_name, 'bundle_item.txt'), 'r') as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

    def load_data_triplets(self, product_type=Product.item):
        with open(os.path.join(self.path, self.dataset_name, f'{str(product_type)}_train_triplets.txt'), 'r') as f:
            return torch.IntTensor(list(map(lambda s: tuple(int(i) for i in s[:-1].split(',')), f.readlines())))

    def triplets_to_csv(self, triplets, product_type):
        with open(os.path.join(self.path, self.dataset_name, f'{str(product_type)}_train_triplets.txt'), 'w') as f:
            f.writelines([f"{','.join([str(i) for i in t])}\n" for t in triplets])

    def pairs_list_to_ground_truth_mtx(self, pairs_list, num_users, num_products):
        indice = np.array(pairs_list, dtype=np.int32)
        values = np.ones(len(pairs_list), dtype=np.float32)
        return sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(num_users, num_products)).tocsr()

    def initiate_train_user_bundle_properties(self, using_triplets_cache=True):
        self.U_B_pairs_train = self.load_U_B_interaction("train")
        self.ground_truth_u_b_train = self.pairs_list_to_ground_truth_mtx(self.U_B_pairs_train, self.num_users, self.num_bundles)
        if not using_triplets_cache:
            df = pd.read_csv(os.path.join(self.path, self.dataset_name, f'user_bundle_train.txt'), delimiter="\t", names=["user", "product"])
            self.bundle_interaction_count = df["product"].value_counts()
            self.bundle_sorted_by_degree = list(self.bundle_interaction_count.index)
            self.num_bundles_with_an_interaction = len(self.bundle_sorted_by_degree)
            self.bundle_to_index = dict([(p, i) for i, p in enumerate(self.bundle_sorted_by_degree)])

    def initiate_test_user_bundle_properties(self, use_mini_test=False):
        U_B_pairs_test = self.load_U_B_interaction("test", use_mini_test)
        self.ground_truth_u_b_all_test = self.pairs_list_to_ground_truth_mtx(U_B_pairs_test, self.num_users, self.num_bundles)
        self.test_relevant_users = list(set([pair[0] for pair in U_B_pairs_test]))
        self.ground_truth_u_b_test_relevant_users = self.ground_truth_u_b_all_test[self.test_relevant_users]
        self.train_mask_only_relevant_test = self.ground_truth_u_b_train[self.test_relevant_users]

    def initiate_tune_user_bundle_properties(self):
        U_B_pairs_tune = self.load_U_B_interaction("tune")
        self.ground_truth_u_b_all_tune = self.pairs_list_to_ground_truth_mtx(U_B_pairs_tune, self.num_users, self.num_bundles)
        self.tune_relevant_users = list(set([pair[0] for pair in U_B_pairs_tune]))
        self.ground_truth_u_b_tune_relevant_users = self.ground_truth_u_b_all_tune[self.tune_relevant_users]
        self.train_mask_only_relevant_tune = self.ground_truth_u_b_train[self.tune_relevant_users]

    def initiate_bundle_items_map(self):
        self.B_I_pairs = self.load_B_I_interaction()
        self.ground_truth_b_i = self.pairs_list_to_ground_truth_mtx(self.B_I_pairs, self.num_bundles, self.num_items)

    def initiate_user_item_properties(self):
        self.U_I_pairs = self.load_U_I_interaction()
        indice = np.array(self.U_I_pairs, dtype=np.int32)
        values = np.ones(len(self.U_I_pairs), dtype=np.float32)
        self.ground_truth_u_i = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_items)).tocsr()

        input_file_name = os.path.join(self.path, self.dataset_name, 'user_item.txt')
        df = pd.read_csv(input_file_name, delimiter="\t", names=["user", "product"])
        self.items_interaction_count = df["product"].value_counts()
        self.items_sorted_by_degree = list(self.items_interaction_count.index)
        self.num_items_with_an_interaction = len(self.items_sorted_by_degree)
        self.item_to_index = dict([(p, i) for i, p in enumerate(self.items_sorted_by_degree)])

    def find_negative_product_by_graph_sampling(self, user, product, product_type=Product.item):
        ground_truth = self.ground_truth_u_i if product_type == Product.item else self.ground_truth_u_b_train
        products_sorted_by_degree = self.items_sorted_by_degree if product_type == Product.item else self.bundle_sorted_by_degree
        num_products_with_interactions = self.num_items_with_an_interaction if product_type == Product.item else self.num_bundles_with_an_interaction
        product_to_index = self.item_to_index if product_type == Product.item else self.bundle_to_index
        product_interaction_count = self.items_interaction_count if product_type == Product.item else self.bundle_interaction_count

        product_index = product_to_index[product]
        product_degree = product_interaction_count[product]

        diff_with_product_below = 1e8
        diff_with_product_above = 1e8
        index_below = product_index - 1
        index_above = product_index + 1

        while index_below >= 0:
            product_below = products_sorted_by_degree[index_below]
            if ground_truth[user, product_below] == 0:
                diff_with_product_below = product_degree - product_interaction_count[product_below]
                break
            index_below -= 1
        while index_above < num_products_with_interactions:
            product_above = products_sorted_by_degree[index_below]
            if ground_truth[user, product_above] == 0:
                diff_with_product_above = product_interaction_count[product_above] - product_degree
                break
            index_above += 1

        if diff_with_product_below == 1e8 and diff_with_product_above == 1e8:
            raise Exception(
                f"Didn't find negative example for user {user} and {str(product_type)} {product} - that's a bug")

        if diff_with_product_above < diff_with_product_below:
            return product_above
        else:
            return product_below

    def create_data_triplets(self, product_type=Product.item):
        triplets = []
        positive_pairs = self.U_I_pairs if product_type == Product.item else self.U_B_pairs_train
        for (user, pos_item) in tqdm(positive_pairs):
            neg_item = self.find_negative_product_by_graph_sampling(user, pos_item, product_type)
            triplets.append((user, pos_item, neg_item))
        return triplets
