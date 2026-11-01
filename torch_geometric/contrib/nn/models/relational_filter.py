from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.contrib.nn.models.hard_concrete import HardConcreteGate
from torch_geometric.typing import NodeType


class RelationalFilter(torch.nn.Module):
    r"""The column- and table-level filtering step of the structural
    optimizer from the `"What Makes a Desired Graph for Relational Deep
    Learning?" <https://arxiv.org/abs/2606.08491>`_ paper.

    Learns which columns of each table, and which whole tables, a relational
    model should use. For a node :math:`v` of type (table) :math:`T` whose
    encoder represents it by one vector :math:`\mathbf{e}_{v,c}` per column
    :math:`c`, the column gates :math:`g_{T,c} \in [0, 1]` scale each column,

    .. math::
        \tilde{\mathbf{e}}_{v,c} = g_{T,c} \, \mathbf{e}_{v,c},

    and the table gate :math:`s_T \in [0, 1]` scales the node's embedding
    :math:`\mathbf{h}_v` before message passing,

    .. math::
        \tilde{\mathbf{h}}_v = s_T \, \mathbf{h}_v.

    All gates are :class:`~torch_geometric.contrib.nn.models.HardConcreteGate`
    switches with an expected :math:`L_0` penalty. In training mode, one
    sample is drawn per gate and call, and shared by all nodes; in evaluation
    mode, the gates are deterministic 0/1 switches.

    The two kinds of gates are applied separately, by :meth:`gate_columns`
    and :meth:`gate_tables`, since the column gates must act inside the
    encoder, before its columns are mixed. :meth:`gate_columns` scales axis 1
    of its input, so it also applies to an input of shape
    :obj:`[num_nodes, channels]`, gating channels instead of columns.

    Args:
        num_columns (Dict[str, int]): The number of column gates per node
            type, *i.e.*, the size of axis 1 of the inputs to
            :meth:`gate_columns`. Node types that are not included get no
            column gates.
        table_types (List[str], optional): The node types that get a table
            gate. Typically all node types except the one being predicted.
            If :obj:`None`, uses the node types in :obj:`num_columns`.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`~torch_geometric.contrib.nn.models.HardConcreteGate`.

    Example:
        >>> filt = RelationalFilter({'users': 3, 'posts': 2},
        ...                         table_types=['posts'])
        >>> e = filt.gate_columns('users', torch.randn(10, 3, 64))
        >>> h_dict = filt.gate_tables({'users': torch.randn(10, 64),
        ...                            'posts': torch.randn(5, 64)})
        >>> column_penalty, table_penalty = filt.l0_penalties()
    """
    def __init__(
        self,
        num_columns: Dict[NodeType, int],
        table_types: Optional[List[NodeType]] = None,
        **kwargs: Any,
    ):
        super().__init__()
        if table_types is None:
            table_types = list(num_columns.keys())

        self.num_columns = dict(num_columns)
        self.table_types = list(table_types)

        # Node types without columns (`0`) get no column gates:
        self.column_gates = torch.nn.ModuleDict({
            node_type:
            HardConcreteGate(num, **kwargs)
            for node_type, num in self.num_columns.items() if num > 0
        })
        self.table_gates = torch.nn.ModuleDict({
            node_type:
            HardConcreteGate(1, **kwargs)
            for node_type in self.table_types
        })

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for gate in self.column_gates.values():
            gate.reset_parameters()
        for gate in self.table_gates.values():
            gate.reset_parameters()

    def gate_columns(self, node_type: NodeType, x: Tensor) -> Tensor:
        r"""Scales axis 1 of :obj:`x` by the column gates of
        :obj:`node_type`.

        Args:
            node_type (str): The node type of :obj:`x`.
            x (torch.Tensor): The column-wise encoding of shape
                :obj:`[num_nodes, num_columns, *]`.

        Returns:
            torch.Tensor: The gated encoding, of the same shape. Returned
            unchanged if :obj:`node_type` has no column gates.
        """
        if node_type not in self.column_gates:
            return x
        if x.size(1) != self.num_columns[node_type]:
            raise ValueError(f"Expected {self.num_columns[node_type]} "
                             f"columns for node type '{node_type}' (got "
                             f"{x.size(1)})")
        gate = self.column_gates[node_type]()
        return x * gate.view(1, -1, *([1] * (x.dim() - 2)))

    def gate_tables(
        self,
        x_dict: Dict[NodeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        r"""Scales the embeddings of every node type by its table gate.

        Args:
            x_dict (Dict[str, torch.Tensor]): The node embeddings of each
                node type, of shape :obj:`[num_nodes, *]`.

        Returns:
            Dict[str, torch.Tensor]: The gated embeddings. Node types without
            a table gate are passed through unchanged.
        """
        return {
            node_type:
            x * self.table_gates[node_type]()
            if node_type in self.table_gates else x
            for node_type, x in x_dict.items()
        }

    def l0_penalties(self) -> Tuple[Tensor, Tensor]:
        r"""Returns the expected number of open column gates
        :math:`\sum_T \sum_c \mathbb{E}\|g_{T,c}\|_0` and of open table gates
        :math:`\sum_T \mathbb{E}\|s_T\|_0`, as differentiable scalars.
        """
        def total(gates: torch.nn.ModuleDict) -> Tensor:
            if len(gates) == 0:
                return torch.zeros(())
            return torch.stack([g.l0_penalty() for g in gates.values()]).sum()

        return total(self.column_gates), total(self.table_gates)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'num_columns={sum(self.num_columns.values())}, '
                f'num_tables={len(self.table_types)})')
