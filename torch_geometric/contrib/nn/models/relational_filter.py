from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from torch_geometric.contrib.nn.models.hard_concrete import HardConcreteGate
from torch_geometric.typing import NodeType


class RelationalFilter(torch.nn.Module):
    r"""The filtering step of the structural optimizer from the
    `"What Makes a Desired Graph for Relational Deep Learning?"
    <https://arxiv.org/abs/2606.08491>`_ paper.

    Sits between a heterogeneous encoder and the GNN, and learns which
    feature channels and which whole node types (tables) the GNN should see.
    For a node :math:`v` of type :math:`T(v)` with encoder output
    :math:`\mathbf{h}^{(0)}_v`,

    .. math::
        \tilde{\mathbf{h}}_v = s_{T(v)} \left( \mathbf{g}_{T(v)} \odot
        \mathbf{h}^{(0)}_v \right),
        \qquad
        \mathbf{z}_v = \boldsymbol{\mu}(\tilde{\mathbf{h}}_v) +
        \boldsymbol{\sigma}(\tilde{\mathbf{h}}_v) \odot \boldsymbol{\varepsilon},
        \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),

    where :math:`\mathbf{g}_T \in [0, 1]^d` are per-feature switches and
    :math:`s_T \in [0, 1]` is a whole-table switch, both
    :class:`~torch_geometric.contrib.nn.models.HardConcreteGate` instances,
    followed by a variational information bottleneck (VIB). At evaluation
    time the switches are hard 0/1 and :math:`\mathbf{z}_v =
    \boldsymbol{\mu}(\tilde{\mathbf{h}}_v)`.

    Args:
        in_channels (int): Size of each input node embedding.
        node_types (List[str]): The node types to filter.
        out_channels (int, optional): Size of each output embedding
            :math:`\mathbf{z}_v`. If :obj:`None`, uses :obj:`in_channels`.
            (default: :obj:`None`)
        feature_gates (bool, optional): If set to :obj:`False`, disables the
            per-feature switches :math:`\mathbf{g}_T`. (default: :obj:`True`)
        table_gates (bool, optional): If set to :obj:`False`, disables the
            whole-table switches :math:`s_T`. (default: :obj:`True`)
        bottleneck (bool, optional): If set to :obj:`False`, disables the
            VIB, so that :math:`\mathbf{z}_v = \tilde{\mathbf{h}}_v`.
            (default: :obj:`True`)
        **gate_kwargs (optional): Additional arguments of
            :class:`~torch_geometric.contrib.nn.models.HardConcreteGate`.

    Example:
        >>> filt = RelationalFilter(64, node_types=['users', 'posts'])
        >>> z_dict = filt({'users': torch.randn(3, 64),
        ...                'posts': torch.randn(5, 64)})
        >>> penalty = (filt.kl_loss() + filt.feature_l0_penalty() +
        ...            filt.table_l0_penalty())
    """
    def __init__(
        self,
        in_channels: int,
        node_types: List[NodeType],
        out_channels: Optional[int] = None,
        feature_gates: bool = True,
        table_gates: bool = True,
        bottleneck: bool = True,
        **gate_kwargs: Any,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.node_types = list(node_types)

        self.feature_gates = torch.nn.ModuleDict({
            node_type: HardConcreteGate(in_channels, **gate_kwargs)
            for node_type in self.node_types
        }) if feature_gates else None
        self.table_gates = torch.nn.ModuleDict({
            node_type: HardConcreteGate(1, **gate_kwargs)
            for node_type in self.node_types
        }) if table_gates else None
        # The VIB encoder outputs mean and log-variance for every node type:
        self.bottleneck = torch.nn.ModuleDict({
            node_type: torch.nn.Linear(in_channels, 2 * self.out_channels)
            for node_type in self.node_types
        }) if bottleneck else None

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for module_dict in [
                self.feature_gates, self.table_gates, self.bottleneck
        ]:
            if module_dict is not None:
                for module in module_dict.values():
                    module.reset_parameters()

    def forward(self, x_dict: Dict[NodeType, Tensor]) -> Dict[NodeType, Tensor]:
        r"""Filters and compresses the node embeddings of every node type.

        Args:
            x_dict (Dict[str, torch.Tensor]): The encoder output
                :math:`\mathbf{h}^{(0)}` for each node type, each of shape
                :obj:`[num_nodes, in_channels]`.

        Returns:
            Dict[str, torch.Tensor]: :math:`\mathbf{z}` for each node type,
            each of shape :obj:`[num_nodes, out_channels]`. Node types not in
            :obj:`node_types` are passed through unchanged.
        """
        raise NotImplementedError

    def kl_loss(self) -> Tensor:
        r"""Returns :math:`\sum_T \overline{\mathrm{KL}}_T` from the last
        forward pass: the KL divergence of :math:`q(\mathbf{z} \mid
        \tilde{\mathbf{h}})` to :math:`\mathcal{N}(\mathbf{0}, \mathbf{I})`,
        averaged over the nodes of each type and summed over types.
        """
        raise NotImplementedError

    def feature_l0_penalty(self) -> Tensor:
        r"""Returns :math:`\sum_T \mathbb{E}\|\mathbf{g}_T\|_0`."""
        raise NotImplementedError

    def table_l0_penalty(self) -> Tensor:
        r"""Returns :math:`\sum_T \mathbb{E}\|s_T\|_0`."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_node_types={len(self.node_types)})')
