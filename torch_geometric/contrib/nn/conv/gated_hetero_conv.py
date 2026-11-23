from typing import Dict, List, Optional

import torch
from torch import Tensor

from torch_geometric.contrib.nn.models.hard_concrete import HardConcreteGate
from torch_geometric.nn.conv import HeteroConv, MessagePassing
from torch_geometric.nn.conv.hetero_conv import group
from torch_geometric.typing import EdgeType, NodeType


class GatedHeteroConv(HeteroConv):
    r"""A :class:`~torch_geometric.nn.conv.HeteroConv` in which some
    relations are scaled by learned switches, from the `"What Makes a Desired
    Graph for Relational Deep Learning?" <https://arxiv.org/abs/2606.08491>`_
    paper (Eqs. 17--18).

    For node :math:`v`, the output sums the relations into :math:`v`'s type,
    each gated relation :math:`k` scaled by its switch
    :math:`g_k \in [0, 1]`:

    .. math::
        \mathbf{z}_v = \sum_{r \in \mathcal{R}} \mathrm{conv}_r(\mathbf{x},
        \mathcal{E}_r)_v + \sum_{k \in \mathcal{K}} g_k \,
        \mathrm{conv}_k(\mathbf{x}, \mathcal{E}_k)_v .

    The switches are
    :class:`~torch_geometric.contrib.nn.models.HardConcreteGate` samples in
    training (one per relation and call) and deterministic 0/1 in
    evaluation; a relation whose switch is 0 is skipped entirely. To share one
    sample across several layers, as the paper's :math:`g_k` has no layer
    index, draw :obj:`gate_values` once with :meth:`sample_gates` and pass
    them to every layer.

    Eq. 17 scales the aggregated neighbor message only; give gated relations
    convolutions without a root weight or bias (*e.g.*,
    :obj:`SAGEConv(..., root_weight=False, bias=False)`) so that their output
    is exactly that message.

    Args:
        convs (Dict[Tuple[str, str, str], MessagePassing]): A convolution per
            edge type, as in :class:`~torch_geometric.nn.conv.HeteroConv`.
        gated_edge_types (List[Tuple[str, str, str]]): The edge types that
            get a switch, in the order of the switches.
        gate (HardConcreteGate, optional): The switches, with one gate per
            entry of :obj:`gated_edge_types`. Pass the same module to several
            layers to share it. If :obj:`None`, a new
            :class:`~torch_geometric.contrib.nn.models.HardConcreteGate` is
            created. (default: :obj:`None`)
        aggr (str, optional): How relations into the same node type are
            combined (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`,
            :obj:`"max"`, :obj:`"cat"`, :obj:`None`). (default: :obj:`"sum"`)
        **kwargs (optional): Arguments of a newly created
            :class:`~torch_geometric.contrib.nn.models.HardConcreteGate`.

    Example:
        >>> conv = GatedHeteroConv({
        ...     ('user', 'buys', 'item'): SAGEConv((-1, -1), 64),
        ...     ('item', 'knn', 'item'): SAGEConv(-1, 64, root_weight=False,
        ...                                       bias=False),
        ... }, gated_edge_types=[('item', 'knn', 'item')])
        >>> out_dict = conv(x_dict, edge_index_dict)
        >>> loss = task_loss + lambda_k * conv.l0_penalty()
    """
    def __init__(
        self,
        convs: Dict[EdgeType, MessagePassing],
        gated_edge_types: List[EdgeType],
        gate: Optional[HardConcreteGate] = None,
        aggr: Optional[str] = 'sum',
        **kwargs,
    ):
        super().__init__(convs, aggr)
        self.gated_edge_types = [tuple(et) for et in gated_edge_types]
        unknown = set(self.gated_edge_types) - set(self.convs.keys())
        if unknown:
            raise ValueError(f"Gated edge types without a convolution: "
                             f"{sorted(unknown)}")
        if len(set(self.gated_edge_types)) != len(self.gated_edge_types):
            raise ValueError("'gated_edge_types' has duplicates")
        if gate is None and len(self.gated_edge_types) > 0:
            gate = HardConcreteGate(len(self.gated_edge_types), **kwargs)
        if gate is not None and gate.num_gates != len(self.gated_edge_types):
            raise ValueError(f"Expected a gate with "
                             f"{len(self.gated_edge_types)} switches (got "
                             f"{gate.num_gates})")
        self.gate = gate
        self._index = {et: i for i, et in enumerate(self.gated_edge_types)}

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        super().reset_parameters()
        if self.gate is not None:
            self.gate.reset_parameters()

    def sample_gates(self) -> Optional[Tensor]:
        r"""Draws the switch values, of shape :obj:`[num_gated]` (random in
        training, 0/1 in evaluation), or :obj:`None` without gated
        relations."""
        return None if self.gate is None else self.gate()

    def open_edge_types(
        self,
        gate_values: Optional[Tensor] = None,
    ) -> List[EdgeType]:
        r"""The gated edge types whose switch is not 0 in
        :obj:`gate_values` (by default, the evaluation-mode switches), so a
        caller can skip building the edges of closed ones (Eq. 18)."""
        if self.gate is None:
            return []
        if gate_values is None:
            gate_values = (self.gate.logit.sigmoid()
                           > self.gate.threshold).to(self.gate.logit)
        return [
            et for et, value in zip(self.gated_edge_types, gate_values)
            if value > 0
        ]

    def l0_penalty(self) -> Tensor:
        r"""The expected number of open switches
        :math:`\sum_k \mathbb{E}\|g_k\|_0`, a differentiable scalar."""
        if self.gate is None:
            return torch.zeros(())
        return self.gate.l0_penalty()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        gate_values: Optional[Tensor] = None,
    ) -> Dict[NodeType, Tensor]:
        r"""Runs the forward pass of the module.

        Args:
            x_dict (Dict[str, torch.Tensor]): The node features of each node
                type.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): The
                edges of each edge type, of shape :obj:`[2, num_edges]`. Edge
                types without a convolution, or missing here, are skipped.
            gate_values (torch.Tensor, optional): The switch values of shape
                :obj:`[num_gated]`, *e.g.*, from :meth:`sample_gates`. If
                :obj:`None`, they are drawn here. (default: :obj:`None`)
        """
        if gate_values is None:
            gate_values = self.sample_gates()
        out_dict: Dict[NodeType, List[Tensor]] = {}
        for edge_type, conv in self.convs.items():
            src, _, dst = edge_type
            if (edge_type not in edge_index_dict or src not in x_dict
                    or dst not in x_dict):
                continue
            index = self._index.get(edge_type)
            if index is not None and not gate_values[index] > 0:
                continue  # A closed switch: the relation is not used.
            if src == dst:
                out = conv(x_dict[src], edge_index_dict[edge_type])
            else:
                out = conv((x_dict[src], x_dict[dst]),
                           edge_index_dict[edge_type])
            if index is not None:
                out = out * gate_values[index]
            out_dict.setdefault(dst, []).append(out)
        return {
            key: group(value, self.aggr)
            for key, value in out_dict.items()
        }

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_relations={len(self.convs)}, '
                f'num_gated={len(self.gated_edge_types)})')
