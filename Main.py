from common.ExperimentsRunner import Experiment
from common.LoggingUtils import create_log_dir
from common.Args import parse_args
from common.ReproducibilityUtils import fix_seed
import os
import logging
import sys
from PostUL import ModelBundleEmbedding as PostULEmbedding
from PostUL import Model as PostULModel
from PreUL import ModelBundleEmbedding as PreULEmbedding
from PreUL import Model as PreULModel
from UserBert import Model as UserBertModel
from UserBert import ModelV2 as UserBertModelV2
from Pretrain import PreTrainer

def main(args, model_class):
    fix_seed(args.seed)
    output_dir, start_time_string = create_log_dir(args)

    # Set logger
    logger = logging.getLogger("Bruce")
    logger.addHandler(logging.StreamHandler(sys.stdout))
    if not args.DontCreateDir:
        logging.basicConfig(filename=os.path.join(output_dir, 'out.log'), level=logging.INFO)
    logger.info(f"CMD parameters: {sys.argv}")
    logger.info(f"Output dir: {output_dir}")

    bruce_experiment = Experiment(args, model_class, output_dir)
    #bruce_experiment.test_baseline()
    if args.test_only:
        bruce_experiment.test_model(output_dir)
    else:
        res_string = bruce_experiment.run_experiment(output_dir, output_dir, start_time_string)
    #recalls.append(Recall)
    #maps.append(MAP)

    #print_map_recall(maps, recalls)
    #save_map_recall(maps, recalls, main_output_dir)


def run_pretrain(args):
    output_dir, start_time_string = create_log_dir(args)
    # Set logger
    logger = logging.getLogger("Bruce")
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logging.basicConfig(filename=os.path.join(output_dir, 'out.log'), level=logging.INFO)
    logger.info(f"CMD parameters: {sys.argv}")
    logger.info(f"Output dir: {output_dir}")

    experiment = PreTrainer.Experiment(args)
    experiment.run_experiment(output_dir)



if __name__ == "__main__":
    args = parse_args(sys.argv)
    if args.run_pretrain:
        run_pretrain(args)
        exit()
    #if args.bundleEmbeddings and args.op_after_transformer == "bert":
    #    raise Exception("Using bundle embedding with bert like op after transformer is not supported, need to "
    #                    "add more ifs to the weights sizes and the attention_mask")
    if args.usePostUL:
        model = PostULEmbedding.Model if args.bundleEmbeddings else PostULModel.Model
    elif args.useUserBert:
        model = UserBertModel.Model
    elif args.useUserBertV2:
        model = UserBertModelV2.Model
    else:
        model = PreULEmbedding.Model if args.bundleEmbeddings else PreULModel.Model
    main(args, model)