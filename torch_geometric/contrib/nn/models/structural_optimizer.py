from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor

from torch_geometric.contrib.nn.models.edge_injection import (
    BehavioralEdges,
    KNNEdges,
    TwoHopEdges,
)
from torch_geometric.contrib.nn.models.hard_concrete import HardConcreteGate
from torch_geometric.contrib.nn.models.relational_filter import (
    RelationalFilter, )
from torch_geometric.contrib.nn.models.variational_bottleneck import (
    VariationalBottleneck, )
from torch_geometric.nn import MLP, LayerNorm, SAGEConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import EdgeType, NodeType


def default_conv(edge_type: EdgeType, channels: int,
                 injected: bool) -> MessagePassing:
    r"""A :class:`~torch_geometric.nn.conv.SAGEConv` per relation; injected
    relations carry neighbor messages only (no root weight or bias)."""
    if injected:
        return SAGEConv((channels, channels), channels, root_weight=False,
                        bias=False)
    return SAGEConv((channels, channels), channels)


class StructuralOptimizer(torch.nn.Module):
    r"""The structural optimizer from the `"What Makes a Desired Graph for
    Relational Deep Learning?" <https://arxiv.org/abs/2606.08491>`_ paper: a
    heterogeneous GNN that learns, together with the task, which tables to
    keep (A), which new relations to add (B, C), and how much of each node
    embedding to pass on (A), with a sparsity penalty (D).

    Given the embeddings :math:`\mathbf{h}^{(0)}_v` of a mini-batch from any
    encoder, it computes, in order:

    1. **Table gates** (Eq. 5): :math:`\tilde{\mathbf{h}}_v = s_T
       \mathbf{h}^{(0)}_v` for every table in :obj:`gated_tables`. Column
       gates act inside the encoder; call :meth:`filter.gate_columns
       <torch_geometric.contrib.nn.models.RelationalFilter.gate_columns>`
       there.
    2. **Injection** (Eqs. 6--8): the per-batch templates add relations
       built from :math:`\tilde{\mathbf{h}}`; dataset-level ones such as
       :class:`~torch_geometric.contrib.transforms.TemporalContinuity` are
       listed in :obj:`injected_edge_types`.
    3. **Bottleneck** (Eq. 14): :math:`\mathbf{z}_v = \boldsymbol{\mu}(
       \tilde{\mathbf{h}}_v) + \boldsymbol{\sigma}(\tilde{\mathbf{h}}_v)
       \odot \boldsymbol{\epsilon}`.
    4. **Gated message passing** (Eqs. 17, 19):
       :class:`~torch_geometric.contrib.nn.conv.GatedHeteroConv` layers,
       each followed by a per-type :class:`~torch_geometric.nn.LayerNorm`
       and ReLU; injected relations are scaled by one switch each, shared by
       all layers. Closed templates are not built.
    5. **Head** (Eq. 20): an :class:`~torch_geometric.nn.MLP` on the target
       nodes.

    :meth:`structural_loss` returns
    :math:`\beta_\text{vib}\,\mathrm{KL} + \lambda \sum \mathbb{E}\|g\|_0 +
    \lambda_s \sum \mathbb{E}\|s\|_0 + \lambda_k \sum \mathbb{E}\|g_k\|_0`
    (Eq. 21) for the last forward pass. With every step off, the model is a
    plain heterogeneous GNN.

    Args:
        node_types (List[str]): The node types.
        edge_types (List[Tuple[str, str, str]]): The schema's relations.
        channels (int): Size of each embedding.
        out_channels (int): Size of each prediction.
        num_layers (int, optional): Number of message passing layers.
            (default: :obj:`2`)
        num_columns (Dict[str, int], optional): Column gates per node type
            (see :class:`~torch_geometric.contrib.nn.models.RelationalFilter`).
            (default: :obj:`None`)
        gated_tables (List[str], optional): Node types with a table gate.
            (default: :obj:`None`)
        templates (List[torch.nn.Module], optional): Per-batch injection
            templates (:class:`~torch_geometric.contrib.nn.models.KNNEdges`,
            :class:`~torch_geometric.contrib.nn.models.BehavioralEdges`,
            :class:`~torch_geometric.contrib.nn.models.TwoHopEdges`).
            (default: :obj:`None`)
        injected_edge_types (List[Tuple[str, str, str]], optional): Injected
            relations already in the graph, *e.g.*, from
            :class:`~torch_geometric.contrib.transforms.TemporalContinuity`.
            (default: :obj:`None`)
        template_gates (bool, optional): Whether injected relations get a
            switch each (step C). (default: :obj:`False`)
        vib (bool, optional): Whether to use the bottleneck.
            (default: :obj:`False`)
        lambda_col (float, optional): Weight :math:`\lambda`.
            (default: :obj:`0.0`)
        lambda_table (float, optional): Weight :math:`\lambda_s`.
            (default: :obj:`0.0`)
        lambda_template (float, optional): Weight :math:`\lambda_k`.
            (default: :obj:`0.0`)
        beta_vib (float, optional): Weight :math:`\beta_\text{vib}`.
            (default: :obj:`0.0`)
        conv (callable, optional): Builds the convolution of a relation from
            :obj:`(edge_type, channels, injected)`; any
            :class:`~torch_geometric.nn.conv.MessagePassing` layer for
            bipartite inputs works, *e.g.*,
            :obj:`lambda et, c, injected: GATConv((c, c), c,
            add_self_loops=False, bias=not injected)`. Injected relations
            should carry the neighbor message only (no root weight or
            bias), so that a closed template adds nothing.
            (default: :func:`default_conv`)
        gate_kwargs (Dict[str, Any], optional): Arguments of every
            :class:`~torch_geometric.contrib.nn.models.HardConcreteGate`.
            (default: :obj:`None`)

    Example:
        >>> model = StructuralOptimizer(
        ...     data.node_types, data.edge_types, channels=64,
        ...     out_channels=1, gated_tables=['races', 'results'],
        ...     templates=[KNNEdges('drivers', k=10)], template_gates=True,
        ...     vib=True, lambda_table=1e-3, lambda_template=1e-3,
        ...     beta_vib=1e-3)
        >>> h_dict = encoder(batch)  # [num_nodes, 64] per node type
        >>> pred = model(h_dict, batch.edge_index_dict, 'drivers',
        ...              batch.batch_dict, batch['drivers'].seed_time,
        ...              num_targets=batch['drivers'].batch_size)
        >>> loss = task_loss(pred, y) + model.structural_loss()
    """
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        out_channels: int,
        num_layers: int = 2,
        num_columns: Optional[Dict[NodeType, int]] = None,
        gated_tables: Optional[List[NodeType]] = None,
        templates: Optional[List[torch.nn.Module]] = None,
        injected_edge_types: Optional[List[EdgeType]] = None,
        template_gates: bool = False,
        vib: bool = False,
        lambda_col: float = 0.0,
        lambda_table: float = 0.0,
        lambda_template: float = 0.0,
        beta_vib: float = 0.0,
        conv: Callable[[EdgeType, int, bool], MessagePassing] = default_conv,
        gate_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        # Imported here: contrib.nn.conv itself imports contrib.nn.models.
        from torch_geometric.contrib.nn.conv import GatedHeteroConv

        gate_kwargs = gate_kwargs or {}
        self.lambda_col = lambda_col
        self.lambda_table = lambda_table
        self.lambda_template = lambda_template
        self.beta_vib = beta_vib

        self.filter = None
        if num_columns or gated_tables:
            self.filter = RelationalFilter(num_columns or {},
                                           table_types=gated_tables or [],
                                           **gate_kwargs)

        self.templates = torch.nn.ModuleList(templates or [])
        injected = list(injected_edge_types or [])
        injected += [t.edge_type for t in self.templates]
        gated = injected if template_gates else []
        gate = HardConcreteGate(len(gated), **gate_kwargs) if gated else None
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            convs = {et: conv(et, channels, False) for et in edge_types}
            convs.update({et: conv(et, channels, True) for et in injected})
            self.convs.append(GatedHeteroConv(convs, gated, gate=gate))
        self.norms = torch.nn.ModuleList([
            ModuleDict(
                {nt: LayerNorm(channels, mode='node')
                 for nt in node_types}) for _ in range(num_layers)
        ])
        self.head = MLP(channels, out_channels=out_channels, norm='batch_norm',
                        num_layers=1)
        self.gate = gate
        self.gated_edge_types = gated
        self.vib = VariationalBottleneck(node_types, channels) if vib else None
        self._kl: Optional[Tensor] = None

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for module in [self.filter, self.vib, self.head, *self.convs]:
            if module is not None:
                module.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def build_templates(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        batch_dict: Optional[Dict[NodeType, Tensor]] = None,
        seed_time: Optional[Tensor] = None,
        only: Optional[set] = None,
    ) -> Dict[EdgeType, Tensor]:
        r"""The edges of every per-batch template, or of those in
        :obj:`only`."""
        num_nodes_dict = {nt: x.size(0) for nt, x in x_dict.items()}
        out = {}
        for template in self.templates:
            if only is not None and template.edge_type not in only:
                continue
            if isinstance(template, KNNEdges):
                edge_type, edge_index = template(x_dict, batch_dict, seed_time)
            elif isinstance(template, BehavioralEdges):
                edge_type, edge_index = template(x_dict, edge_index_dict,
                                                 batch_dict, seed_time)
            elif isinstance(template, TwoHopEdges):
                edge_type, edge_index = template(edge_index_dict,
                                                 num_nodes_dict)
            else:
                raise ValueError(f"Unknown template {template}")
            out[edge_type] = edge_index
        return out

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        target_type: NodeType,
        batch_dict: Optional[Dict[NodeType, Tensor]] = None,
        seed_time: Optional[Tensor] = None,
        num_targets: Optional[int] = None,
    ) -> Tensor:
        r"""Predicts for the first :obj:`num_targets` nodes of
        :obj:`target_type`.

        Args:
            x_dict (Dict[str, torch.Tensor]): The encoder's embeddings
                :math:`\mathbf{h}^{(0)}` per node type, of shape
                :obj:`[num_nodes, channels]`.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): The
                mini-batch connectivity.
            target_type (str): The node type to predict.
            batch_dict (Dict[str, torch.Tensor], optional): The subgraph of
                each node (temporal sampling); with :obj:`seed_time`, keeps
                injected edges from reaching into the past of a newer seed.
                (default: :obj:`None`)
            seed_time (torch.Tensor, optional): The time of each subgraph's
                seed. (default: :obj:`None`)
            num_targets (int, optional): The number of target nodes (the
                first ones). If :obj:`None`, all. (default: :obj:`None`)

        Returns:
            torch.Tensor: Predictions of shape
            :obj:`[num_targets, out_channels]`.
        """
        if self.filter is not None:
            x_dict = self.filter.gate_tables(x_dict)
        gates = None if self.gate is None else self.gate()
        if len(self.templates) > 0:
            only = None if gates is None else {
                et
                for et, g in zip(self.gated_edge_types, gates) if g > 0
            }
            injected = self.build_templates(x_dict, edge_index_dict,
                                            batch_dict, seed_time, only)
            edge_index_dict = {**edge_index_dict, **injected}
        self._kl = None
        if self.vib is not None:
            x_dict, self._kl = self.vib(x_dict)
        for conv, norm_dict in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict, gates)
            x_dict = {
                key: norm_dict[key](x).relu()
                for key, x in x_dict.items()
            }
        x = x_dict[target_type]
        return self.head(x if num_targets is None else x[:num_targets])

    def structural_loss(self) -> Tensor:
        r"""The penalty of Eq. 21 for the last forward pass, a
        differentiable scalar (0 with every step off)."""
        loss = self.head.lins[0].weight.new_zeros(())
        if self.filter is not None:
            col, table = self.filter.l0_penalties()
            loss = loss + self.lambda_col * col + self.lambda_table * table
        if self.gate is not None:
            loss = loss + self.lambda_template * self.gate.l0_penalty()
        if self._kl is not None:
            loss = loss + self.beta_vib * self._kl
        return loss

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_layers={len(self.convs)}, '
                f'filter={self.filter is not None}, '
                f'templates={len(self.templates)}, '
                f'template_gates={self.gate is not None}, '
                f'vib={self.vib is not None})')
