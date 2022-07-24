import torch
import numpy as np


def get_is_hit(scores, ground_truth, topk):
    device = scores.device
    values, col_indice = torch.topk(scores, topk)
    row_indice = torch.zeros_like(col_indice) + torch.arange(
        scores.shape[0], device=device, dtype=torch.long).view(-1, 1)
    is_hit = ground_truth[row_indice.view(-1),
                          col_indice.view(-1)].view(-1, topk)
    return is_hit, col_indice


class _Metric:
    """
    base class of metrics like Recall@k NDCG@k MRR@k
    """

    def __init__(self):
        self.start()

    @property
    def metric(self):
        return self._metric

    def __call__(self, scores, ground_truth, is_hit):
        """
        - scores: model output
        - ground_truth: one-hot test dataset shape=(users, all_bundles/all_items).
        """
        raise NotImplementedError

    def get_title(self):
        raise NotImplementedError

    def start(self):
        """
        clear all
        """
        self._cnt = 0
        self._metric = 0
        self._sum = 0

    def stop(self):
        self._metric = self._sum/self._cnt


class Recall(_Metric):
    """
    Recall in top-k samples
    They calculate TP@K / TP+FN @K
    """

    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        self.epsilon = 1e-8

    def get_title(self):
        return "Recall@{}".format(self.topk)

    # scores, ground truth shapes = (4096, 22864) = (#users, #bundles)
    def __call__(self, scores, ground_truth, is_hit):
        # TODO: if we want to optimize performance we should take is_hit outside and pass it as a parameter
        new_is_hit = is_hit.sum(dim=1)
        #num_pos = ground_truth.sum(dim=1)
        # New recall
        num_pos = ground_truth.sum(dim=1).clamp(0, self.topk).to(torch.long)
        count = scores.shape[0] - (num_pos == 0).sum().item()
        score = (new_is_hit / (num_pos + self.epsilon)).sum().item()
        self._cnt += count
        self._sum += score
        return score/count


class NDCG(_Metric):
    """
    NDCG in top-k samples
    In this work, NDCG = log(2)/log(1+hit_positions)
    """
    # Looks good
    # hit is an array of 0/1 for whether the bundle was bought by the user (relevance)
    # the lower part is log{2..k+2} - log of the rank of each item + 1
    # in this case the 2 versions of ndcg are the same (2^rel_i -1) vs (rel_i)
    def DCG(self, hit, device=torch.device('cpu')):
        hit = hit/torch.log2(torch.arange(2, self.topk+2,
                                          device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(self, num_pos):
        hit = torch.zeros(self.topk, dtype=torch.float)
        hit[:num_pos] = 1
        return self.DCG(hit)

    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        self.IDCGs = torch.empty(1 + self.topk, dtype=torch.float)
        self.IDCGs[0] = 1  # avoid 0/0
        for i in range(1, self.topk + 1):
            self.IDCGs[i] = self.IDCG(i)

    def get_title(self):
        return "NDCG@{}".format(self.topk)

    def __call__(self, scores, ground_truth, is_hit):
        device = scores.device
        num_pos = ground_truth.sum(dim=1).clamp(0, self.topk).to(torch.long)
        dcg = self.DCG(is_hit, device)
        idcg = self.IDCGs[num_pos]
        ndcg = dcg/idcg.to(device)
        count = scores.shape[0] - (num_pos == 0).sum().item()
        score = ndcg.sum().item()
        self._cnt += count
        self._sum += score
        return score/count


class MRR(_Metric):
    """
    Mean reciprocal rank in top-k samples
    """

    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        self.denominator = torch.arange(1, self.topk+1, dtype=torch.float)

    def get_title(self):
        return "MRR@{}".format(self.topk)

    def __call__(self, scores, ground_truth, is_hit):
        device = scores.device
        # We divide each binary is_hit with its position in the ranking
        new_is_hit = is_hit / self.denominator.to(device)
        first_hit_rr = new_is_hit.max(dim=1)[0]
        num_pos = ground_truth.sum(dim=1)
        # We add the count only items that had any positive item
        count = scores.shape[0] - (num_pos == 0).sum().item()
        score = first_hit_rr.sum().item()
        self._cnt += count
        self._sum += score
        return score/count


class MAP(_Metric):
    """
    Mean reciprocal rank in top-k samples
    """

    def __init__(self, topk):
        super().__init__()
        self.topk = topk

    def get_title(self):
        return "MAP@{}".format(self.topk)

    def calc_average_precision(self, is_hit, num_pos, device):
        num_users = is_hit.shape[0]
        k = is_hit.shape[1]
        precision_sum = torch.zeros(num_users).to(device)
        current_tp = torch.zeros(num_users).to(device)
        is_hit_transpose = is_hit.transpose(0, 1).to(device)
        for rank in range(k):
            is_tp = is_hit_transpose[rank]  # shaped: num_users
            current_tp += is_tp
            precision_sum += is_tp * (current_tp / (rank + 1))
        # need to test if this line is the same as the next:
        return precision_sum / num_pos.clamp(1, self.topk).to(torch.float)

    def __call__(self, scores, ground_truth, is_hit):
        device = scores.device
        num_pos = ground_truth.sum(dim=1)
        average_precision = self.calc_average_precision(is_hit, num_pos, device)
        count = scores.shape[0] - (num_pos == 0).sum().item()
        score = average_precision.sum().item()
        self._cnt += count
        self._sum += score
        return score/count
