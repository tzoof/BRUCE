import torch
from common.LoggingUtils import *
import pickle
import random
import logging

logger = logging.getLogger("Bruce")


def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_random_state():
    logger.info("random state")
    logger.info(random.random())
    logger.info(random.getstate())
    logger.info(torch.get_rng_state())
    logger.info(np.random.get_state())
    logger.info(np.random.random())


def save_state(out_dir):
    with open(os.path.join(out_dir, "random_state"), 'wb') as random_state:
        pickle.dump(random.getstate(), random_state)
    with open(os.path.join(out_dir, "torch_state"), 'wb') as torch_state:
        pickle.dump(torch.get_rng_state(), torch_state)
    with open(os.path.join(out_dir, "numpy_state"), 'wb') as numpy_state:
        pickle.dump(np.random.get_state(), numpy_state)


def load_state(in_dir):
    logger.info(f"Loading random states from {in_dir}")
    with open(os.path.join(in_dir, "random_state"), 'rb') as random_state:
        state = pickle.load(random_state)
        random.setstate(state)
    with open(os.path.join(in_dir, "torch_state"), 'rb') as torch_state:
        torch.set_rng_state(pickle.load(torch_state))
    with open(os.path.join(in_dir, "numpy_state"), 'rb') as numpy_state:
        np.random.set_state(pickle.load(numpy_state))