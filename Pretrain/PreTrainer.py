import pickle
import logging
import copy
import math
import sys
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from common.LoggingUtils import *
from sklearn.metrics import auc, precision_score, recall_score, roc_auc_score
from common.Dataset import Dataset
from common.DataLoader import ItemItemDatasetForPretrain
from Pretrain.CooccurenceModel import Model
from common.ReproducibilityUtils import *
from common.LoggingUtils import create_log_dir, draw_pretrain_loss
from common.Args import parse_args

torch.set_printoptions(profile="full")
logger = logging.getLogger("Bruce")


class Experiment():
    def __init__(self, args, output_dir):
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")
        self.args = args
        self.num_items = self.get_num_items(self.args.dataset_string)
        self.item_padding_id = self.num_items
        self.model = Model(self.args, self.num_items).to(self.device)
        self.model = nn.DataParallel(self.model)

        # If loading from an existing model
        if args.model_path != "":
            logger.info(f"Loading model from model_path_to_load")
            self.model.load_state_dict(torch.load(args.model_path))
            # If the model to load contains saved state
            model_to_load_dir = os.path.dirname(args.model_path)
            if os.path.exists(os.path.join(model_to_load_dir, "random_state")) and args.load_random_state:
                load_state(model_to_load_dir)
            logger.info("Initialize real train dataset with new params")

        train_set = ItemItemDatasetForPretrain(args.dataset_string, "train")
        val_set = ItemItemDatasetForPretrain(args.dataset_string, "val")
        test_set = ItemItemDatasetForPretrain(args.dataset_string, "test")
        self.trainDataLoader = DataLoader(train_set, args.pretrain_train_batch_size, shuffle=True)
        self.valDataLoader = DataLoader(val_set, args.pretrain_train_batch_size, shuffle=True)
        self.testDataLoader = DataLoader(test_set, args.pretrain_test_batch_size, shuffle=True)

    def run_experiment(self):
        self.model.train()

        start_time = datetime.now()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(lr=self.args.lr, weight_decay=self.args.weight_decay,
                                         params=self.model.parameters())
        #optimizer = torch.optim.SGD(lr=self.args.lr, weight_decay=self.args.weight_decay,
        #                                 params=self.model.parameters())
        losses = []
        best_epoch = 0
        best_validation_loss = 1000000
        consecutive_loss_increase = 0
        iterations = 0
        for epoch in range(self.args.num_epochs):
            epoch_losses = []
            for batch, (items, labels) in tqdm(enumerate(self.trainDataLoader)):
                predictions = self.model(items).squeeze()
                loss = criterion(predictions, labels)
                losses.append(loss)
                epoch_losses.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iterations += 1
                if (iterations % self.args.evaluate_every == 0 and iterations > self.args.start_val_from):
                    current_val_loss = self.evaluate_on_val(criterion)
                    logger.info(f"iteration: {iterations}\ttrain loss: {loss}\tvalidation loss: {current_val_loss}")
                    # Early stopping
                    if current_val_loss < best_validation_loss:
                        best_validation_loss = current_val_loss
                        best_epoch = epoch
                        consecutive_loss_increase = 0
                    else:
                        consecutive_loss_increase += 1
                    if consecutive_loss_increase >= 2:
                        break

            logger.info(f"epoch: {epoch}, train loss: {sum(epoch_losses)/len(epoch_losses)}")
            if consecutive_loss_increase >= 2:
                break

        time_after_train = datetime.now()
        logger.info(f"Training time: {time_after_train - start_time}")
        logger.info(f"finished training, epoch: {epoch}")
        draw_pretrain_loss(losses, output_dir)
        # Testing best model
        logger.info("Running best model on test set")
        results = self.test(f"{self.output_dir}/pretrain/raw_test_results.tsv")
        with open(f"{self.output_dir}/pretrain/test_results.tsv", 'w') as w:
            w.write(str(results))
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "pretrain", "best_model.pth"))
        logger.info(f"Test time: {datetime.now() - time_after_train}")
        logger.info(f"Output dir: {output_dir}")

    def get_num_items(self, dataset_name):
        with open(os.path.join("data", dataset_name, f'{dataset_name}_data_size.txt'), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][2]

    def evaluate_on_val(self, criterion):
        self.model.eval()
        losses = []
        for batch, (items, labels) in enumerate(self.valDataLoader):
            predictions = self.model(items).squeeze()
            loss = criterion(predictions, labels)
            losses.append(loss)
        self.model.train()
        return sum(losses)/len(losses)

    def to01(self, score):
        return 1 if score > 0.5 else 0

    def test(self, raw_results_file_path):
        logger.info("Test")
        self.model.eval()
        all_preds = []
        all_labels = []
        with open(raw_results_file_path, 'w') as output_file:
            for epoch, (items, labels) in tqdm(enumerate(self.testDataLoader)):
                predictions = self.model(items)
                predictions_as_bool = [self.to01(score) for score in predictions]
                all_preds.append(predictions_as_bool)
                all_labels.append(labels.tolist())
                #for ids, float_pred, label in zip(items, predictions, labels):
                #    output_file.write(f"{ids}\t{float_pred}\t{label}\n")

        all_preds = self.flatten(all_preds)
        all_labels = self.flatten(all_labels)
        results = {
            "precision": round(precision_score(all_labels, all_preds), 4),
            "recall": round(recall_score(all_labels, all_preds), 4),
            "auc": round(roc_auc_score(all_labels, all_preds), 4),
        }
        logger.info(results)
        self.model.train()
        return results

    def flatten(self, list_of_lists):
        return [x for listt in list_of_lists for x in listt]


if __name__ == "__main__":
    # Best args for Steam: --dataset_string=Steam --description=ready --run_pretrain --lr=1e-3 --weight_decay=1e-5 --evaluate_every=100 --start_val_from=-1 --num_epochs=300 --pretrain_train_batch_size=2048
    # Best args for Youshu: --description=debug --run_pretrain --lr=1e-3 --weight_decay=1e-5 --evaluate_every=500 --start_val_from=-1 --num_epochs=1 --pretrain_train_batch_size=2048
    # Best args for NetEase: --dataset_string=NetEase --description=ready --run_pretrain --lr=1e-3 --weight_decay=1e-5 --evaluate_every=500 --start_val_from=-1 --num_epochs=50 --pretrain_train_batch_size=2048

    args = parse_args(sys.argv)
    output_dir, start_time_string = create_log_dir(args)
    # Set logger
    logger = logging.getLogger("Bruce")
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logging.basicConfig(filename=os.path.join(output_dir, 'out.log'), level=logging.INFO)
    logger.info(f"CMD parameters: {sys.argv}")
    logger.info(f"Output dir: {output_dir}")

    experiment = Experiment(args, output_dir)
    experiment.run_experiment()
