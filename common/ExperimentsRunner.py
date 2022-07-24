import pickle
import logging
import copy
import math
import sys
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
from transformers import AdamW, WarmupLinearSchedule
from common.LoggingUtils import *
from common.Dataset import Dataset
from common.Metrics import Recall, NDCG, MRR, MAP, get_is_hit
from common.DataLoader import BundlesTrainDataSet, ItemsTrainDataSet, BundlesRepresentationDataset, get_bundles_representation
from common.ReproducibilityUtils import *
from Pretrain.CooccurenceModel import Model as PretrainingModel

torch.set_printoptions(profile="full")
logger = logging.getLogger("Bruce")


class Experiment():
    def __init__(self, args, model_class, sub_dir_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")
        self.args = args
        self.dataset = Dataset(path='Data', args=args, use_mini_test=args.use_mini_test)
        self.item_padding_id = self.dataset.num_items
        self.cls_token = self.dataset.num_items + 1
        self.bundles_representation, self.attention_masks = get_bundles_representation(self.dataset, self.args, self.item_padding_id, self.cls_token, self.device)
        bundles_representation_dataset = BundlesRepresentationDataset(self.bundles_representation, self.attention_masks)
        self.bundles_representation_loader = DataLoader(bundles_representation_dataset, args.test_batch_size, shuffle=False)
        self.model = model_class(self.args, self.dataset.num_users, self.dataset.num_items, self.dataset.num_bundles).to(self.device)
        self.model = nn.DataParallel(self.model)

        if args.model_path != "":
            logger.info(f"Loading model from {args.model_path}")
            self.model.load_state_dict(torch.load(args.model_path))
            # If the model to load contains saved state
            model_to_load_dir = os.path.dirname(args.model_path)
            if os.path.exists(os.path.join(model_to_load_dir, "random_state")) and args.load_random_state:
                load_state(model_to_load_dir)
                # print_random_state()
            else:
                logger.info("Initialize dummy train datasets with previous params to achieve the same random state")
                items_train_dataset = ItemsTrainDataSet(self.dataset, self.args.num_epochs_on_previous_train * self.args.batch_size)
                bundles_train_dataset = BundlesTrainDataSet(self.dataset, self.args.num_epochs_on_previous_train * self.args.batch_size)
                #print_random_state()

            logger.info("Initialize real train dataset with new params")
            items_train_dataset = ItemsTrainDataSet(self.dataset, self.args.num_epochs * self.args.batch_size)
            bundles_train_dataset = BundlesTrainDataSet(self.dataset, self.args.num_epochs * self.args.batch_size)
        else:
            if args.use_pretrained:
                self.load_embeddings_from_path()
            items_train_dataset = self.load_and_cache_dataset("item", sub_dir_name)
            bundles_train_dataset = self.load_and_cache_dataset("bundle", sub_dir_name)

        self.bundles_train_loader = DataLoader(bundles_train_dataset, args.batch_size, shuffle=False)
        self.items_train_loader = DataLoader(items_train_dataset, args.batch_size, shuffle=False)

    def run_experiment(self, sub_dir_name, main_output_dir, start_time_string):
        # print_random_state()
        self.model.train()
        start_time = datetime.now()
        if self.args.apply_warmup:
            total_training_steps = self.args.num_epochs if self.args.dont_multi_task else self.args.num_epochs * 2 # for one step per task, divide by 2 for one step per batch

            embeddings = [x for name,x in self.model.named_parameters() if name in ["module.item_embeddings.weight", "module.user_embeddings.weight"]]
            params_other_than_embeddings = [x for name,x in self.model.named_parameters() if name not in ["module.item_embeddings.weight", "module.user_embeddings.weight"]]
            optimizer = torch.optim.AdamW([{'lr':self.args.lr, 'weight_decay':0, 'params':embeddings, 'eps':self.args.adam_epsilon, 'correct_bias':False},
                {'lr':self.args.lr, 'weight_decay':self.args.weight_decay, 'params':params_other_than_embeddings, 'eps':self.args.adam_epsilon, 'correct_bias':False}])

            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.args.warmup_steps,
                                             t_total=total_training_steps)
        else:
            embeddings = [x for name,x in self.model.named_parameters() if name in ["module.item_embeddings.weight", "module.user_embeddings.weight"]]
            params_other_than_embeddings = [x for name,x in self.model.named_parameters() if name not in ["module.item_embeddings.weight", "module.user_embeddings.weight"]]
            optimizer = torch.optim.Adam([{'lr':self.args.lr, 'weight_decay':0, 'params':embeddings},
                {'lr':self.args.lr, 'weight_decay':self.args.weight_decay, 'params':params_other_than_embeddings}])

        item_losses = []
        bundle_losses = []
        # logger.info("Initial test")
        # res_string, output_metrics, raw_results = self.get_hit()
        best_model = None
        best_model_metrics = None
        best_res_string = ""
        best_recall_at_5 = -1
        best_epoch = 0
        validation_ran = False
        print(f"Multi task?: {not self.args.dont_multi_task}")
        for epoch, (items, bundles) in tqdm(enumerate(zip(self.items_train_loader, self.bundles_train_loader))):
            if not self.args.dont_multi_task:
                item_loss = self.get_item_loss(items)
                item_losses.append(item_loss)
                optimizer.zero_grad()
                item_loss.backward()
                optimizer.step()
                if self.args.apply_warmup:
                    scheduler.step()

            bundle_loss = self.get_bundle_loss(bundles)

            bundle_losses.append(bundle_loss)
            optimizer.zero_grad()
            bundle_loss.backward()
            optimizer.step()
            if self.args.apply_warmup:
                scheduler.step()

            if (epoch % self.args.evaluate_every == 0):
                if self.args.dont_multi_task:
                    logger.info(f"epoch: {epoch}, bundle loss: {sum(bundle_losses[-100:]) / min(epoch + 1, 100)}")
                    # logger.info(f"epoch: {epoch}, bundle loss: {bundle_loss}")
                else:
                    logger.info(
                        f"epoch: {epoch}, item loss: {sum(item_losses[-100:]) / min(epoch + 1, 100)}, bundle loss: {sum(bundle_losses[-100:]) / min(epoch + 1, 100)}")
                    # logger.info(f"epoch: {epoch}, item loss: {item_loss}, bundle loss: {bundle_loss}")

            if ((epoch % self.args.evaluate_every == 0 or epoch == self.args.num_epochs - 1) and epoch > self.args.start_val_from ):
                validation_ran = True
                logger.info(f"epoch: {epoch}, starting test")
                res_string, output_metrics, raw_results = self.get_hit("tune")
                if output_metrics[self.args.checkpoint_metric_index].metric > best_recall_at_5:
                    best_res_string = res_string

                    best_model = copy.deepcopy(self.model)
                    best_epoch = epoch
                    best_recall_at_5 = output_metrics[self.args.checkpoint_metric_index].metric
            if epoch >= self.args.end_training_at:
                # Stop the training - this is so I can use the same dataset (6000) for several experiments but make their last 2000 epochs with the validation paralel
                break
            if (epoch % 500 == 0 and epoch > 0 and epoch > self.args.start_val_from):
                if best_model is not None:
                    logger.info(
                        f"Temporal checkpoint: epoch {epoch} saving best model at : {sub_dir_name}/model_{epoch}.pth")
                    torch.save(best_model.state_dict(), os.path.join(sub_dir_name, f"model_best_{best_epoch}_epoch_{epoch}.pth"))

        time_after_train = datetime.now()
        logger.info(f"Training time: {time_after_train - start_time}")
        logger.info(f"finished training, epoch: {epoch}")

        # Testing best model
        logger.info("Saving latest model")
        if not self.args.DontCreateDir:
            torch.save(self.model.state_dict(), os.path.join(sub_dir_name, "latest_model.pth"))
        if validation_ran and self.args.dont_test_last_epoch:
            logger.info("Running latest model on test set")
            res_string, output_metrics, raw_results = self.get_hit("test")

        if validation_ran:
            self.model = best_model

        log_results_string(sub_dir_name, best_res_string, "tune")
        if best_model is not None:
            logger.info(f"Best epoch according to metric {self.args.checkpoint_metric_index} is {best_epoch}")
        else:
            logger.info(f"No validation, Running only on last epoch")

        logger.info("Best results on validation (tune) set")
        logger.info(best_res_string)
        logger.info("Running best model on test set")

        test_res_string, output_metrics, raw_results = self.get_hit("test")
        log_results_string(sub_dir_name, test_res_string, "test")
        log_raw_results(sub_dir_name, raw_results, "test")
        draw_item_loss(item_losses, sub_dir_name)
        draw_bundle_loss(bundle_losses, sub_dir_name)
        logger.info(f"Eval time: {datetime.now() - time_after_train}")
        logger.info(f"Saving model to dir: {sub_dir_name}")
        if not self.args.DontCreateDir and best_model is not None:
            torch.save(best_model.state_dict(), os.path.join(sub_dir_name, "best_model.pth"))
        logger.info(
            f"{self.args.dataset_string}	{start_time_string}	{main_output_dir}	{self.args.model_name}	{self.args.max_bundle_size}	{self.args.bundle_selection_string}	\
        {self.args.embedding_string}	{self.args.sampling_string}	{self.args.num_epochs}	{self.args.batch_size}	{time_after_train - start_time}	{test_res_string}")
        logger.info(f"Best epoch according to metric {self.args.checkpoint_metric_index} is {best_epoch}")
        return test_res_string

    def load_embeddings_from_path(self):
        # Loading items embeddings from pretrained co-occurence NN
        if self.args.pretrained_path != "":
            logger.info(f"Loading pretrained embeddings from {self.args.pretrained_path}")
            pretrained_state_dict = torch.load(self.args.pretrained_path)
            my_state_dict = self.model.state_dict()
            for name, param in pretrained_state_dict.items():
                if "embedding" in name:
                    if isinstance(param, Parameter):
                        # backwards compatibility for serialized parameters
                        param = param.data
                    my_state_dict[name].copy_(param)
        # Loading users and items embeddings from pretrained bpr
        elif self.args.pretrained_bpr_path != "":
            logger.info(f"Loading pretrained embeddings from bpr model at {self.args.pretrained_bpr_path}")
            with open(self.args.pretrained_bpr_path, 'rb') as pkl:
                bpr_model = pickle.load(pkl)
                my_state_dict = self.model.state_dict()
                num_items_in_embedding =  my_state_dict["module.item_embeddings.weight"].shape[0]
                num_users_in_embedding = my_state_dict["module.user_embeddings.weight"].shape[0]

                bpr_users_weights = bpr_model.user_factors / bpr_model.user_norms.reshape(-1, 1)
                bpr_items_weights = bpr_model.item_factors / bpr_model.item_norms.reshape(-1, 1)

                # Adding dummy users and items - for padding and more (initializing them)
                added_items_vectors = np.random.rand(self.args.embed_shape, num_items_in_embedding - bpr_items_weights.shape[0])
                added_users_vectors = np.random.rand(self.args.embed_shape, num_users_in_embedding - bpr_users_weights.shape[0])

                bpr_items_weights = np.concatenate((bpr_items_weights, added_items_vectors.transpose()))
                bpr_users_weights = np.concatenate((bpr_users_weights, added_users_vectors.transpose()))

                my_state_dict["module.item_embeddings.weight"].copy_(torch.Tensor(bpr_items_weights))
                my_state_dict["module.user_embeddings.weight"].copy_(torch.Tensor(bpr_users_weights))
        else:
            raise ValueError("You must pass the pretrained_path or pretrained_bpr_path when using use_pretrained argument")


    def load_and_cache_dataset(self, type, sub_dir_name):
        file_name = os.path.join("Data", self.args.dataset_string, f"train_{type}_dataset_epochs{self.args.num_epochs}_batch{self.args.batch_size}_seed{self.args.seed}")
        if os.path.exists(file_name) and not self.args.overwrite_cache:
            logger.info(f"Loading data from cached file {file_name}")
            train_dataset = torch.load(file_name)
        else:
            if type == "bundle":
                logger.info(f"Creating bundles train dataset")
                train_dataset = BundlesTrainDataSet(self.dataset, self.args.num_epochs * self.args.batch_size)
            else:
                logger.info(f"Creating items train dataset")
                train_dataset = ItemsTrainDataSet(self.dataset, self.args.num_epochs * self.args.batch_size)
            logger.info(f"Saving train dataset to file {file_name}")
            torch.save(train_dataset, file_name)
            save_state(sub_dir_name)
        return train_dataset

    def test_model(self, sub_dir_name):
        if self.args.model_path == "":
            raise Exception("You are in test_only mode but did not pass passed --model_path")
        # self.model = model_class(self.args, self.dataset.num_users, self.dataset.num_items).to(self.device)
        self.model.eval()

        ks_to_save_is_hit_for = [20, 40, 80]
        is_hit_files_per_k = {}
        for k in ks_to_save_is_hit_for:
            file_name = os.path.join(sub_dir_name, f"bruce_correctBundlesAt{k}_test.txt")
            is_hit_files_per_k[k] = open(file_name, 'w')

        # raw_results_file_path = get_raw_results_file_path(sub_dir_name, "test")
        results_per_user_file = open(get_per_user_results_file_path(sub_dir_name, "test"), 'w')
        res_string, output_metrics, _ = self.get_hit("test", results_per_user_file=results_per_user_file, is_hit_files_per_k=is_hit_files_per_k)
        log_results_string(sub_dir_name, res_string, "test")

        # closing all the files I opened:
        for k in is_hit_files_per_k:
            is_hit_files_per_k[k].close()
        results_per_user_file.close()

        return res_string

    def pred_by_rank_all_bundles(self):
        list_of_degrees_per_bundle = [0] * self.dataset.num_bundles
        for bundle_index, degree in self.dataset.bundle_interaction_count.items():
            list_of_degrees_per_bundle[bundle_index] = degree
        return torch.tensor(list_of_degrees_per_bundle, dtype=torch.float).reshape(1, -1)

    def safe_write(self, file, string):
        if file is not None:
            file.write(string)

    def test_one_at_a_time_less_memory(self, testOrTune="tune", raw_results_file=None, results_per_user_file=None, is_hit_files_per_k={}):
        '''
        test for dot-based model
        '''
        raw_results = []
        logger.info("Measuring")
        if testOrTune == "tune":
            relevant_users = self.dataset.tune_relevant_users
            ground_truth_u_b_test_relevant_users = self.dataset.ground_truth_u_b_tune_relevant_users
            train_mask_only_relevant = self.dataset.train_mask_only_relevant_tune
        else:
            relevant_users = self.dataset.test_relevant_users
            ground_truth_u_b_test_relevant_users = self.dataset.ground_truth_u_b_test_relevant_users
            train_mask_only_relevant = self.dataset.train_mask_only_relevant_test

        self.model.eval()
        metrics = [Recall(5), Recall(10), Recall(20), Recall(40), Recall(80),
                   MAP(5), MAP(10), MAP(20), MAP(40), MAP(80),
                   MRR(5), MRR(10), MRR(20), MRR(40), MRR(80),
                   NDCG(5), NDCG(10), NDCG(20), NDCG(40), NDCG(80)]

        ks = set([metric.topk for metric in metrics])
        for metric in metrics:
            metric.start()
        start = datetime.now()
        if results_per_user_file is not None:
            results_per_user_file.write("\t".join(["user"] + [metric.get_title() for metric in metrics]) + "\n")
        for k in is_hit_files_per_k:
            is_hit_files_per_k[k].write("user\tCorrectBundles\n")
        with torch.no_grad():
            for i in tqdm(range(len(relevant_users))):
                user = torch.tensor([relevant_users[i]]).reshape(1, -1)

                all_prediction_scores = []
                for bundle_indices, bundles_representation, attention_masks in self.bundles_representation_loader:
                    prediction_scores = self.pred_bundles_1_user_for_test(user, bundle_indices, bundles_representation, attention_masks)
                    all_prediction_scores.append(prediction_scores)
                    if testOrTune == "test" and raw_results_file is not None:
                        current_raw_results = [(user, bundle_id, score.item()) for (bundle_id, score) in zip(bundle_indices, prediction_scores)]
                        for res in current_raw_results:
                            raw_results_file.write(f"{res}\n")

                    # raw_results.extend([(user, bundle_id, score.item()) for (bundle_id, score) in zip(bundle_indices, prediction_scores)])
                # The assertions are to verify nothing get missed in parallel run
                assert len(all_prediction_scores) == math.ceil(self.dataset.num_bundles/self.args.test_batch_size), f"len: {len(all_prediction_scores)} num_bundles; {self.dataset.num_bundles}, batch size: {self.args.test_batch_size}"
                prediction_scores = torch.cat(all_prediction_scores, dim=0).reshape(1, -1)
                assert prediction_scores.shape == (1, self.dataset.num_bundles), f"shape: {prediction_scores.shape} num_bundles; {self.dataset.num_bundles}"

                test_ground_truth_u_b_only_relevant = torch.from_numpy(
                    ground_truth_u_b_test_relevant_users[i].toarray()).reshape(1, -1)
                train_mask_u_b_only_relevant = torch.from_numpy(
                    train_mask_only_relevant[i].toarray()).reshape(1, -1)
                prediction_scores -= 1e8 * train_mask_u_b_only_relevant.to(self.device)
                ground_truth = test_ground_truth_u_b_only_relevant.to(self.device)
                is_hit_atk = {}
                top_bundles = {}
                self.safe_write(results_per_user_file, f"{user.item()}\t")

                for k in ks:
                    is_hit_atk[k], top_bundles[k] = get_is_hit(prediction_scores, ground_truth, k)
                for metric in metrics:
                    user_metric_score = metric(prediction_scores, ground_truth, is_hit_atk[metric.topk])
                    self.safe_write(results_per_user_file, f"\t{round(user_metric_score, 4)}")
                for k in is_hit_files_per_k:
                    try:
                        top_bundles_k = top_bundles[k]
                        flatten_is_hit = is_hit_atk[k].nonzero(as_tuple=True)[1].tolist()
                        correct_top_bundles = top_bundles_k[0][flatten_is_hit]
                        self.safe_write(is_hit_files_per_k[k], f"{user.item()}\t{str(correct_top_bundles.tolist())}\n")
                    except:
                        print("bye")
                self.safe_write(results_per_user_file, "\n")

            logger.info(f"Test: time={datetime.now() - start}")
            for metric in metrics:
                metric.stop()
            self.model.train()
            return metrics, raw_results

    def test_baseline(self):
        '''
        test degree based model - giving each bundle its degree in train
        '''
        logger.info("Testing baseline")
        self.model.eval()
        metrics = [Recall(5), Recall(10), Recall(20), Recall(40), Recall(80),
                   MAP(5), MAP(10), MAP(20), MAP(40), MAP(80),
                   MRR(5), MRR(10), MRR(20), MRR(40), MRR(80),
                   NDCG(5), NDCG(10), NDCG(20), NDCG(40), NDCG(80)]
        ks = set([metric.topk for metric in metrics])
        for metric in metrics:
            metric.start()
        start = datetime.now()
        with torch.no_grad():
            # Doing one prediction since its the same for all
            prediction_scores = self.pred_by_rank_all_bundles().reshape(1, self.dataset.num_bundles).to(self.device)
            for i in range(len(self.dataset.test_relevant_users)):
                test_ground_truth_u_b_only_relevant = torch.from_numpy(
                    self.dataset.ground_truth_u_b_test_relevant_users[i].toarray()).reshape(1, -1)
                train_mask_u_b_only_relevant = torch.from_numpy(
                    self.dataset.train_mask_only_relevant_test[i].toarray()).reshape(1, -1)
                prediction_scores -= (1e8 * train_mask_u_b_only_relevant).to(self.device)
                ground_truth = test_ground_truth_u_b_only_relevant.to(self.device)
                is_hit_atk = {}
                for k in ks:
                    is_hit_atk[k] = get_is_hit(prediction_scores, ground_truth, k)
                for metric in metrics:
                    metric(prediction_scores, ground_truth, is_hit_atk[metric.topk])
            logger.info(f"Test: time={datetime.now() - start}")
            for metric in metrics:
                metric.stop()
            res_string = ""
            for metric in metrics:
                res_string += f"{metric.get_title()}: {round(metric.metric, 6)}\t"
            print(res_string)
            self.model.train()
            return metrics

    def pred_bundles_1_user_for_test(self, users_ids, bundle_ids, bundles_representation, attention_masks=None):
        repeated_users = users_ids.repeat_interleave(bundles_representation.shape[0]).to(device=self.device,
                                                                                         dtype=torch.long)
        return self.model(bundles_representation, repeated_users, bundle_ids.to(dtype=torch.long), bundle=True, attention_masks=attention_masks)

    def get_hit(self, testOrTune="tune", raw_results_file=None, results_per_user_file=None, is_hit_files_per_k={}):
        self.model.eval()
        if self.args.test_like_BCGN:
            output_metrics, raw_results = self.test_one_at_a_time_less_memory(testOrTune, raw_results_file, results_per_user_file, is_hit_files_per_k)
            res_string = ""
            for metric in output_metrics:
                res_string += f"{metric.get_title()}: {round(metric.metric, 6)}\t"
        else:
            Recall, MAP = self.measure_like_DAM(5)
            res_string = f"Recall: {Recall}, MAP: {MAP}"
        self.model.train()
        logger.info(res_string)
        return res_string, output_metrics, raw_results

    def pred_bundle(self, users_ids, bundles_ids):
        '''
        users_ids, bundles_ids shape is (batch_size,)
        '''
        bundles_representation = self.bundles_representation[bundles_ids.to(dtype=torch.long)]
        users_representation = users_ids.to(dtype=torch.long, device=self.device).unsqueeze(1)
        attention_masks = self.attention_masks[bundles_ids.to(dtype=torch.long)]
        return self.model(bundles_representation, users_representation, bundles_ids.to(dtype=torch.long), bundle=True, attention_masks=attention_masks)


    def get_bundle_loss(self, bundles):
        u, pb, nb = bundles
        upb, user_regularization, pos_bundle_regularization = self.pred_bundle(u, pb)
        unb, user_regularization, neg_bundle_regularization = self.pred_bundle(u, nb)
        return -torch.sum(torch.log(torch.sigmoid(upb - unb))) + self.args.weight_decay * (user_regularization + pos_bundle_regularization + neg_bundle_regularization)

    def get_item_loss(self, items):
        u, p, n = items
        p = p.reshape(self.args.batch_size, 1)
        n = n.reshape(self.args.batch_size, 1)
        padding_size = self.args.max_bundle_size - 1
        p = nn.functional.pad(p, (0, padding_size), mode='constant', value=self.item_padding_id).to(self.device,
                                                                                                    dtype=torch.long)
        n = nn.functional.pad(n, (0, padding_size), mode='constant', value=self.item_padding_id).to(self.device,
                                                                                                    dtype=torch.long)
        u = u.to(self.device, dtype=torch.long)
        p, user_regularization, pos_bundle_regularization = self.model(p, u)
        n, user_regularization, neg_bundle_regularization = self.model(n, u)
        return -torch.sum(torch.log(torch.sigmoid(p - n))) + self.args.weight_decay * (user_regularization + pos_bundle_regularization + neg_bundle_regularization)

    def is_rng_unchanged(self):
        return (self.rng_state_before_test == torch.get_rng_state()).all()
