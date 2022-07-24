""" Created using the example in: https://github.com/benfred/implicit """

import argparse
import codecs
import logging
import time
import os
from datetime import datetime
import pickle
from tqdm import tqdm
import numpy as np
from common.Metrics import Recall, NDCG, MRR, MAP
import torch

from implicit.bpr import BayesianPersonalizedRanking
from common.Dataset import Dataset

def create_out_dir(args):
    start_time_string = datetime.now().strftime("%Y-%m-%d-%H%M")
    bundle = "avg_items" if args.avg_items else "bundle" 
    dirname = os.path.join("TrainedModels", f"bpr_user_{bundle}_{args.dataset_string}_{start_time_string}")
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    return dirname


def items_recommendations_to_bundle_recommendations(items_recommendations, dataset):
    item_score_dict = to_dict(items_recommendations)
    bundle_scores = []
    for bundle in range(dataset.num_bundles):
        items_in_bundle = dataset.ground_truth_b_i[bundle].nonzero()[1]
        if len(items_in_bundle) == 0:
            bundle_scores.append((bundle,0))
            continue
        bundle_items_scores = [item_score_dict[item] for item in items_in_bundle]

        bundle_scores.append((bundle, sum(bundle_items_scores) / len(bundle_items_scores)))
    return sorted(bundle_scores, key=lambda x:x[1], reverse=True)


def to_dict(list_tuples):
    new_dict = {}
    for a,b in list_tuples:
        new_dict[a] = b
    return new_dict


def get_is_hit(sorted_predictions_scores, user_ground_truth, topk):
    top_k_bundle_ids = [tup[0] for tup in sorted_predictions_scores[:topk]]
    is_hit_k = [user_ground_truth[0, bundle] for bundle in top_k_bundle_ids]
    #return torch.Tensor((is_hit_k)
    return torch.Tensor(is_hit_k).reshape(1,-1)

    

def test(model, output_dir, dataset, train_interactions):
    metrics = [Recall(5), Recall(10), Recall(20), Recall(40), Recall(80),
                MAP(5), MAP(10), MAP(20), MAP(40), MAP(80),
                MRR(5), MRR(10), MRR(20), MRR(40), MRR(80),
                NDCG(5), NDCG(10), NDCG(20), NDCG(40), NDCG(80)]
                
    ks = set([metric.topk for metric in metrics])
    for metric in metrics:
        metric.start()

    # Header  
    start = datetime.now()
    results_per_user_file = open(os.path.join(output_dir, f"metrics_per_user.tsv"), 'w')
    results_per_user_file.write("\t".join(["user"] + [metric.get_title() for metric in metrics]) + "\n")

    #predictions_dict = {}
    relevant_users = dataset.test_relevant_users
    #relevant_users = range(2)

    for userid in tqdm(relevant_users):
        is_hit_atk = {}
        k = dataset.num_items if args.avg_items else dataset.num_bundles 
        recommendations = model.recommend(userid, train_interactions, N=k, filter_already_liked_items=not args.avg_items)
        if args.avg_items:
            recommendations = items_recommendations_to_bundle_recommendations(recommendations, dataset)
        #predictions_dict[userid] = recommendations

        ground_truth = torch.from_numpy(dataset.ground_truth_u_b_all_test[userid].toarray()).reshape(1, -1)

        num_pos_user_items = ground_truth.sum(dim=1).item()
        if num_pos_user_items != 0:
        #ground_truth = dataset.ground_truth_u_b_all_test[userid]
            results_per_user_file.write(str(userid))
            for k in ks:
                is_hit_atk[k] = get_is_hit(recommendations, ground_truth, k)
            for metric in metrics:
                scores_mock = torch.zeros((1, dataset.num_bundles))
                user_metric_score = metric(scores_mock, ground_truth, is_hit_atk[metric.topk])
                results_per_user_file.write(f"\t{round(user_metric_score, 4)}")
            results_per_user_file.write("\n")
    
    for metric in metrics:
        metric.stop()
    results_per_user_file.close()
    return metrics


def calculate_recommendations(args):
    """ Generates bundle recommendations for each user in the dataset """
    # Create a model from the input data
    model = BayesianPersonalizedRanking(factors=args.size - 1, verify_negative_samples=True, iterations=100)
    dataset = Dataset(path='Data', args=args, use_mini_test=False)
    train_interactions = dataset.ground_truth_u_i if args.avg_items else dataset.ground_truth_u_b_train

    output_dir = create_out_dir(args)
    if args.model_to_load != "":
        with open(args.model_to_load, 'rb') as pkl:
            model = pickle.load(pkl)
    else:
        logging.debug("Training bpr model")
        start = time.time()
        model.fit(train_interactions.T.tocsr(), show_progress=True)
        pickle.dump(model, open(os.path.join(output_dir, "model.pkl"), 'wb'))
        logging.debug("Trained bpr model in %0.2fs saved to %s", time.time() - start, os.path.join(output_dir, "model.pkl"))

    if not args.train_only:
        logging.debug("Generating recommendations")
        results_tsv = os.path.join(output_dir, f"recommendations_{start_time_string}.tsv")
        # generate recommendations for each user and write out to a file
        start = time.time()
        metrics = test(model, output_dir, dataset, train_interactions)

        result_string = "\t".join([f"{metric.get_title()}: {round(metric.metric, 4)}" for metric in metrics])
        print(result_string)

        with open(os.path.join(output_dir, "final_metrics.txt"), 'w') as final_res:
            final_res.write(result_string)

        logging.debug("generated recommendations in %0.2fs", time.time() - start)
        with open(os.path.join(output_dir, "results.pkl"), 'wb') as p:
            pickle.dump(results_dict, p)


if __name__ == "__main__":
    start_time_string = datetime.now().strftime("%Y-%m-%d-%H%M")
    parser = argparse.ArgumentParser(
        description="bpr recommender",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument(
        "--avg_items",
        type=bool,
        default=False,
        dest="avg_items",
        help="avg_items",
    )
    parser.add_argument(
        "--train_only",
        default=False,
        dest="train_only",
        action='store_true'
    )
    parser.add_argument(
        "--dataset_string",
        type=str,
        default="Steam",
        dest="dataset_string",
        help="dataset_string, should be Steam, Youshu, or NetEase"
    )

    parser.add_argument(
        "--model_to_load",
        type=str,
        default="",
        dest="model_to_load",
        help="model_to_load",
    )
    parser.add_argument(
        "--use_graph_sampling",
        type=bool,
        default=False,
        dest="use_graph_sampling",
        help="use_graph_sampling",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="64",
        dest="size",
        help="size of each item/bundle/user vector",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)
    calculate_recommendations(args)
