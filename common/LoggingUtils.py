import os
import shutil
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt


def create_log_dir(args):
    if args.DontCreateDir:
        return None, ""
    dataset_string = args.dataset_string
    desc = args.description
    test_only = "test_only" if args.test_only else ""
    pretrain = "pretrain" if args.run_pretrain else ""
    start_time_string = datetime.now().strftime("%Y-%m-%d-%H%M")
    dir_name = os.path.join("TrainedModels", f"{dataset_string}_{start_time_string}_{desc}_{test_only}{pretrain}")
    os.mkdir(dir_name)
    shutil.copytree("common", os.path.join(dir_name, "common"))
    shutil.copytree("UserBert", os.path.join(dir_name, "UserBert"))
    shutil.copytree("PostUL", os.path.join(dir_name, "PostUL"))
    shutil.copytree("PostUL", os.path.join(dir_name, "Pretrain"))
    shutil.copytree("PreUL", os.path.join(dir_name, "PreUL"))
    shutil.copy("Main.py", os.path.join(dir_name, "Main.py"))
    return dir_name, start_time_string


def get_raw_results_file_path(output_dir, TestOrTune="test"):
    return os.path.join(output_dir, f"raw_results_{TestOrTune}.txt")


def get_per_user_results_file_path(output_dir, TestOrTune="test"):
    return os.path.join(output_dir, f"results_per_user_{TestOrTune}.txt")


def log_results_string(output_dir, results_string, TestOrTune="tune"):
    with open(os.path.join(output_dir, f"results_string_{TestOrTune}.txt"), 'w') as res_file:
        res_file.write(results_string)


def log_raw_results(output_dir, raw_results, TestOrTune="tune"):
    if output_dir is None:
        return
    with open(os.path.join(output_dir, f"raw_results_{TestOrTune}.txt"), 'w') as res_file:
        for res in raw_results:
            res_file.write(f"{res}\n")


def save_map_recall(maps, recalls, output_dir):
    with open(os.path.join(output_dir, "maps_and_recall.txt"), 'w') as outfile:
        outfile.write(f"Avg recall: {sum(recalls) / len(recalls)}\n")
        outfile.write(f"Recall std: {np.std(recalls)}\n")
        outfile.write(f"Avg map: {sum(maps) / len(maps)}\n")
        outfile.write(f"Map std: {np.std(maps)}\n")

        outfile.write("Recalls:\n")
        outfile.write(str(recalls))
        outfile.write("\nMaps:\n")
        outfile.write(str(maps))


def print_map_recall(maps, recalls):
    logger.info(f"Avg recall: {sum(recalls) / len(recalls)}")
    logger.info(f"Recall std: {np.std(recalls)}")
    logger.info(f"Avg map: {sum(maps) / len(maps)}")
    logger.info(f"Map std: {np.std(maps)}")

    logger.info("Recalls:")
    logger.info(recalls)
    logger.info("Maps:")
    logger.info(maps)


def draw_pretrain_loss(losses, output_dir, epoch=0):
    if output_dir is None:
        return
    plt.plot(losses)
    plt.title("item-item loss")
    plt.savefig(os.path.join(output_dir, f"item-item-loss-epoch-{epoch}.png"))
    plt.show()


def draw_item_loss(item_losses, output_dir, epoch=0):
    if output_dir is None:
        return
    plt.plot(item_losses)
    plt.title("user-item loss")
    plt.savefig(os.path.join(output_dir, f"user-item-loss-epoch-{epoch}.png"))
    # plt.show()


def draw_bundle_loss(bundle_losses, output_dir, epoch=0):
    if output_dir is None:
        return
    plt.plot(bundle_losses)
    plt.title("user-bundle loss")
    plt.savefig(os.path.join(output_dir, f"user-bundle-loss-epoch-{epoch}.png"))
    # plt.show()
