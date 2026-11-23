from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import NodeType


class VariationalBottleneck(torch.nn.Module):
    r"""The type-wise variational information bottleneck from the `"What
    Makes a Desired Graph for Relational Deep Learning?"
    <https://arxiv.org/abs/2606.08491>`_ paper (Eqs. 14 and 21), after
    `"Deep Variational Information Bottleneck"
    <https://arxiv.org/abs/1612.00410>`_.

    Each node :math:`v` of type :math:`T` is mapped to a stochastic latent

    .. math::
        \mathbf{z}_v = \boldsymbol{\mu}_T(\tilde{\mathbf{h}}_v) +
        \boldsymbol{\sigma}_T(\tilde{\mathbf{h}}_v) \odot
        \boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim
        \mathcal{N}(\mathbf{0}, \mathbf{I}),

    with :math:`\boldsymbol{z}_v = \boldsymbol{\mu}_T(\tilde{\mathbf{h}}_v)`
    in evaluation mode. :math:`\boldsymbol{\mu}_T` and
    :math:`\log \boldsymbol{\sigma}^2_T` are linear maps per node type. The
    penalty is the KL divergence to the prior, averaged over the nodes of
    each type and summed over types:

    .. math::
        \sum_{T} \frac{1}{|T|} \sum_{v \in T}
        \mathrm{KL}\left( q(\mathbf{z}_v \mid \tilde{\mathbf{h}}_v)
        \,\|\, \mathcal{N}(\mathbf{0}, \mathbf{I}) \right), \quad
        \mathrm{KL} = \frac{1}{2} \sum_j \left( \mu_j^2 + \sigma_j^2 -
        \log \sigma_j^2 - 1 \right).

    Args:
        node_types (List[str]): The node types to compress. Others are passed
            through unchanged.
        in_channels (int): Size of each input embedding.
        out_channels (int, optional): Size of each latent. If :obj:`None`,
            equal to :obj:`in_channels`. (default: :obj:`None`)

    Example:
        >>> vib = VariationalBottleneck(['users', 'items'], 64)
        >>> z_dict, kl = vib(h_dict)
        >>> loss = task_loss + beta_vib * kl
    """
    def __init__(
        self,
        node_types: List[NodeType],
        in_channels: int,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mu = ModuleDict({
            node_type:
            torch.nn.Linear(in_channels, out_channels)
            for node_type in node_types
        })
        self.logvar = ModuleDict({
            node_type:
            torch.nn.Linear(in_channels, out_channels)
            for node_type in node_types
        })

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for lin in [*self.mu.values(), *self.logvar.values()]:
            lin.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
    ) -> Tuple[Dict[NodeType, Tensor], Tensor]:
        r"""Compresses the embeddings of every node type it was built for.

        Args:
            x_dict (Dict[str, torch.Tensor]): The embeddings
                :math:`\tilde{\mathbf{h}}` of each node type, of shape
                :obj:`[num_nodes, in_channels]`.

        Returns:
            (Dict[str, torch.Tensor], torch.Tensor): The latents of shape
            :obj:`[num_nodes, out_channels]` (other node types unchanged),
            and the KL penalty as a scalar.
        """
        out, kl = dict(x_dict), None
        for node_type, x in x_dict.items():
            if node_type not in self.mu:
                continue
            mu = self.mu[node_type](x)
            logvar = self.logvar[node_type](x)
            if self.training:
                eps = torch.randn_like(mu)
                out[node_type] = mu + (0.5 * logvar).exp() * eps
            else:
                out[node_type] = mu
            if x.size(0) > 0:  # Mean over the nodes of this type:
                term = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(-1)
                kl = term.mean() if kl is None else kl + term.mean()
        if kl is None:
            some = next(iter(self.mu.values())).weight
            kl = some.new_zeros(())
        return out, kl

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_node_types={len(self.mu)}, '
                f'{self.in_channels}, {self.out_channels})')
