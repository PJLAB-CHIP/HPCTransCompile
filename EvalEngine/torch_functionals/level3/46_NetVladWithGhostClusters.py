import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th


def module_fn(
    x: torch.Tensor,
    clusters: torch.Tensor,
    clusters2: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_mean: torch.Tensor,
    bn_var: torch.Tensor,
    is_training: bool,
    cluster_size: int,
    feature_size: int,
) -> torch.Tensor:
    """
    Functional version of the NetVLAD with ghost clusters

    Args:
        x: Input tensor of shape (batch_size, num_features, feature_size)
        clusters: Weight tensor for cluster assignments
        clusters2: Weight tensor for visual words
        bn_weight: BatchNorm weight
        bn_bias: BatchNorm bias
        bn_mean: BatchNorm running mean
        bn_var: BatchNorm running var
        is_training: Whether in training mode
        cluster_size: Number of clusters (K)
        feature_size: Feature dimension (D)

    Returns:
        Output tensor of shape (batch_size, cluster_size * feature_size)
    """
    max_sample = x.size()[1]
    x = x.view(-1, feature_size)  # B x N x D -> BN x D

    if x.device != clusters.device:
        msg = f"x.device {x.device} != cluster.device {clusters.device}"
        raise ValueError(msg)

    assignment = th.matmul(x, clusters)  # (BN x D) x (D x (K+G)) -> BN x (K+G)
    assignment = F.batch_norm(
        assignment, bn_mean, bn_var, bn_weight, bn_bias, is_training
    )

    assignment = F.softmax(assignment, dim=1)  # BN x (K+G) -> BN x (K+G)
    # remove ghost assigments
    assignment = assignment[:, :cluster_size]
    assignment = assignment.view(-1, max_sample, cluster_size)  # -> B x N x K
    a_sum = th.sum(assignment, dim=1, keepdim=True)  # B x N x K -> B x 1 x K
    a = a_sum * clusters2

    assignment = assignment.transpose(1, 2)  # B x N x K -> B x K x N

    x = x.view(-1, max_sample, feature_size)  # BN x D -> B x N x D
    vlad = th.matmul(assignment, x)  # (B x K x N) x (B x N x D) -> B x K x D
    vlad = vlad.transpose(1, 2)  # -> B x D x K
    vlad = vlad - a

    # L2 intra norm
    vlad = F.normalize(vlad)

    # flattening + L2 norm
    vlad = vlad.reshape(-1, cluster_size * feature_size)  # -> B x DK
    vlad = F.normalize(vlad)
    return vlad  # B x DK


class Model(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(Model, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = 1 / math.sqrt(feature_size)
        clusters = cluster_size + ghost_clusters

        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))

        # Extract batchnorm parameters
        bn = nn.BatchNorm1d(clusters)
        self.bn_weight = nn.Parameter(bn.weight.data.clone())
        self.bn_bias = nn.Parameter(bn.bias.data.clone())
        self.bn_mean = nn.Parameter(bn.running_mean.data.clone())
        self.bn_var = nn.Parameter(bn.running_var.data.clone())

        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.clusters,
            self.clusters2,
            self.bn_weight,
            self.bn_bias,
            self.bn_mean,
            self.bn_var,
            self.training,
            self.cluster_size,
            self.feature_size,
        )


batch_size = 32
num_features = 100
num_clusters = 32
feature_size = 512
ghost_clusters = 16


def get_inputs():
    return [torch.randn(batch_size, num_features, feature_size)]


def get_init_inputs():
    return [num_clusters, feature_size, ghost_clusters]
