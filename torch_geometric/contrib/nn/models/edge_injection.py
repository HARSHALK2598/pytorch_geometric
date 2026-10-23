from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.typing import EdgeType, NodeType


class KNNEdges(torch.nn.Module):
    r"""The *KNN* injection template of the `"What Makes a Desired Graph for
    Relational Deep Learning?" <https://arxiv.org/abs/2606.08491>`_ paper.

    Within a mini-batch, links every node :math:`v` of :obj:`node_type` to the
    :math:`k` nodes :math:`u` of the same type whose encoder embeddings
    :math:`\mathbf{h}^{(0)}` are most cosine-similar to its own. Only nodes
    that are not newer than :math:`v` are candidates
    (:math:`t_u \leq t_v`, :math:`u \neq v`), so no future information leaks
    into :math:`v`. The edges are returned under the edge type
    :obj:`(node_type, 'knn', node_type)`, ready to be gated by
    :class:`~torch_geometric.contrib.nn.conv.GatedHeteroConv`.

    Uses :meth:`torch_geometric.nn.pool.knn` when :obj:`torch-cluster` is
    installed, and an exact pure-PyTorch search otherwise.

    Args:
        node_type (str): The node type to connect.
        k (int, optional): The number of neighbours :math:`n` per node.
            (default: :obj:`10`)
        time_attr (str, optional): The key of the node timestamps in
            :obj:`time_dict`. If no timestamps are passed, all other nodes are
            candidates. (default: :obj:`"time"`)

    Example:
        >>> knn = KNNEdges('drivers', k=5)
        >>> edge_type, edge_index = knn(x_dict, time_dict)
        >>> edge_index_dict[edge_type] = edge_index
    """
    def __init__(self, node_type: NodeType, k: int = 10,
                 time_attr: str = 'time'):
        super().__init__()
        self.node_type = node_type
        self.k = k
        self.time_attr = time_attr

    @property
    def edge_type(self) -> EdgeType:
        return (self.node_type, 'knn', self.node_type)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        time_dict: Optional[Dict[NodeType, Tensor]] = None,
    ) -> Tuple[EdgeType, Tensor]:
        r"""Computes the KNN edges of the current mini-batch.

        Args:
            x_dict (Dict[str, torch.Tensor]): Node embeddings for each node
                type.
            time_dict (Dict[str, torch.Tensor], optional): Node timestamps for
                each node type. (default: :obj:`None`)

        Returns:
            (Tuple[str, str, str], torch.Tensor): The new edge type and its
            :obj:`edge_index` of shape :obj:`[2, num_edges]` (source
            :math:`u` in row 0, target :math:`v` in row 1).
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.node_type!r}, k={self.k})'


class BehavioralEdges(torch.nn.Module):
    r"""The *behavioural* injection template of the `"What Makes a Desired
    Graph for Relational Deep Learning?" <https://arxiv.org/abs/2606.08491>`_
    paper.

    Two nodes :math:`u, v` of :obj:`node_type` are linked if they behave
    alike, *i.e.*, if the mean embeddings of their neighbours along the
    relation :obj:`via` are similar:

    .. math::
        u \to v \quad \text{if} \quad \cos\left(
        \bar{\mathbf{h}}_{\mathcal{N}_B(u)},
        \bar{\mathbf{h}}_{\mathcal{N}_B(v)} \right) > \tau_{\text{sim}},
        \qquad \bar{\mathbf{h}}_{\mathcal{N}_B(\cdot)} =
        \mathrm{mean}_{w \in \mathcal{N}_B(\cdot)} \mathbf{h}^{(0)}_w .

    As in :class:`KNNEdges`, only nodes that are not newer than :math:`v` are
    candidates, and at most :obj:`max_neighbors` edges point into each node.
    The edges are returned under the edge type
    :obj:`(node_type, 'behavioral', node_type)`.

    Args:
        node_type (str): The node type to connect.
        via (Tuple[str, str, str]): The relation :math:`B` whose neighbours
            describe a node's behaviour, as an edge type
            :obj:`(other, rel, node_type)`.
        threshold (float, optional): The similarity threshold
            :math:`\tau_{\text{sim}}`. (default: :obj:`0.9`)
        max_neighbors (int, optional): The maximum number of incoming edges
            per node, keeping the most similar ones. (default: :obj:`10`)
        time_attr (str, optional): The key of the node timestamps in
            :obj:`time_dict`. (default: :obj:`"time"`)
    """
    def __init__(
        self,
        node_type: NodeType,
        via: EdgeType,
        threshold: float = 0.9,
        max_neighbors: int = 10,
        time_attr: str = 'time',
    ):
        super().__init__()
        if via[-1] != node_type:
            raise ValueError(f"'via' must point into '{node_type}' "
                             f"(got {via})")
        self.node_type = node_type
        self.via = tuple(via)
        self.threshold = threshold
        self.max_neighbors = max_neighbors
        self.time_attr = time_attr

    @property
    def edge_type(self) -> EdgeType:
        return (self.node_type, 'behavioral', self.node_type)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        time_dict: Optional[Dict[NodeType, Tensor]] = None,
    ) -> Tuple[EdgeType, Tensor]:
        r"""Computes the behavioural edges of the current mini-batch.

        Args:
            x_dict (Dict[str, torch.Tensor]): Node embeddings for each node
                type.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): The
                mini-batch connectivity; must contain :obj:`via`.
            time_dict (Dict[str, torch.Tensor], optional): Node timestamps for
                each node type. (default: :obj:`None`)

        Returns:
            (Tuple[str, str, str], torch.Tensor): The new edge type and its
            :obj:`edge_index` of shape :obj:`[2, num_edges]`.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.node_type!r}, '
                f'via={self.via}, threshold={self.threshold})')
