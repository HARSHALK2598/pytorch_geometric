from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import coalesce, scatter

TimeArgs = Tuple[Optional[Tensor], Optional[Tensor]]


def node_seed_time(
    batch: Optional[Tensor],
    seed_time: Optional[Tensor],
) -> Optional[Tensor]:
    r"""The seed time of the subgraph each node belongs to.

    With temporal neighbor sampling, a mini-batch holds one subgraph per seed,
    sampled with times up to that seed's time :math:`t_s`. :obj:`batch` gives
    each node's subgraph (as in :obj:`data[node_type].batch`), and
    :obj:`seed_time` the time of each subgraph.

    Returns:
        torch.Tensor or None: :obj:`seed_time[batch]` of shape
        :obj:`[num_nodes]`, or :obj:`None` if either is :obj:`None`.
    """
    if batch is None or seed_time is None:
        return None
    return seed_time[batch]


@torch.no_grad()
def similar_edges(
    x: Tensor,
    num_neighbors: int,
    threshold: Optional[float] = None,
    node_time: Optional[Tensor] = None,
    chunk_size: int = 2048,
) -> Tensor:
    r"""Links every node :math:`v` to the :obj:`num_neighbors` nodes
    :math:`u \neq v` whose rows of :obj:`x` are most cosine-similar to its
    own, as edges :math:`u \to v`.

    Only nodes that cannot leak the future are candidates: if
    :obj:`node_time` is given, :math:`u` is a candidate for :math:`v` only if
    :math:`t_u \le t_v`. If :obj:`threshold` is given, only similarities
    above it count (so a node may get fewer neighbors).

    Similarities are computed :obj:`chunk_size` target nodes at a time, so
    memory stays at :obj:`chunk_size * num_nodes` similarities. No gradient
    flows through the choice of neighbors.

    Args:
        x (torch.Tensor): Node features of shape :obj:`[num_nodes, channels]`.
        num_neighbors (int): The maximum number of incoming edges per node.
        threshold (float, optional): The similarity a candidate must exceed.
            (default: :obj:`None`)
        node_time (torch.Tensor, optional): Node times of shape
            :obj:`[num_nodes]`. (default: :obj:`None`)
        chunk_size (int, optional): The number of target nodes per chunk.
            (default: :obj:`2048`)

    Returns:
        torch.Tensor: The edges of shape :obj:`[2, num_edges]` (sources in
        row 0, targets in row 1).
    """
    num_nodes = x.size(0)
    k = min(num_neighbors, num_nodes - 1)
    if k <= 0:
        return x.new_empty((2, 0), dtype=torch.long)

    x = F.normalize(x, p=2, dim=-1)
    sources, targets = [], []
    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)
        target = torch.arange(start, end, device=x.device)
        sim = x[start:end] @ x.t()  # [chunk, num_nodes]
        sim[target - start, target] = float('-inf')  # No self-loops.
        if node_time is not None:  # No source newer than its target:
            newer = node_time.view(1, -1) > node_time[start:end].view(-1, 1)
            sim.masked_fill_(newer, float('-inf'))
        if threshold is not None:
            sim.masked_fill_(sim <= threshold, float('-inf'))
        value, index = sim.topk(k, dim=1)
        keep = value > float('-inf')
        sources.append(index[keep])
        targets.append(target.view(-1, 1).expand_as(index)[keep])
    return torch.stack([torch.cat(sources), torch.cat(targets)], dim=0)


class KNNEdges(torch.nn.Module):
    r"""The *type-wise KNN* injection template from the `"What Makes a
    Desired Graph for Relational Deep Learning?"
    <https://arxiv.org/abs/2606.08491>`_ paper (Eq. 6).

    Within a mini-batch, links every node :math:`v` of :obj:`node_type` to
    the :math:`n` nodes :math:`u` of the same type whose embeddings
    :math:`\mathbf{h}^{(0)}` are most cosine-similar to its own:

    .. math::
        \mathcal{E}_{\text{knn}} = \left\{ (u, v) \;\middle|\; u \in
        \operatorname{Top-}n_{u' \in \mathcal{C}(v)} \cos\left(
        \mathbf{h}^{(0)}_{u'}, \mathbf{h}^{(0)}_v \right) \right\},

    where the candidates :math:`\mathcal{C}(v)` exclude :math:`v` itself and,
    with temporal sampling, every node from a subgraph whose seed is newer
    than :math:`v`'s (:math:`t_s(u) > t_s(v)`), so that no information from
    the future reaches :math:`v`. Each node *receives* from its :math:`n`
    most similar candidates. The edges are returned under the edge type
    :obj:`(node_type, 'knn', node_type)`, to be used as an extra relation,
    *e.g.*, in :class:`~torch_geometric.nn.conv.HeteroConv`.

    Args:
        node_type (str): The node type to connect.
        k (int, optional): The number of neighbors :math:`n` per node.
            (default: :obj:`10`)
        chunk_size (int, optional): The number of nodes whose similarities
            are computed at once (see :func:`similar_edges`).
            (default: :obj:`2048`)

    Example:
        >>> knn = KNNEdges('drivers', k=5)
        >>> edge_type, edge_index = knn(x_dict, batch.batch_dict,
        ...                             batch['drivers'].seed_time)
        >>> edge_index_dict[edge_type] = edge_index
    """
    def __init__(self, node_type: NodeType, k: int = 10,
                 chunk_size: int = 2048):
        super().__init__()
        self.node_type = node_type
        self.k = k
        self.chunk_size = chunk_size

    @property
    def edge_type(self) -> EdgeType:
        r"""The edge type of the added edges."""
        return (self.node_type, 'knn', self.node_type)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        batch_dict: Optional[Dict[NodeType, Tensor]] = None,
        seed_time: Optional[Tensor] = None,
    ) -> Tuple[EdgeType, Tensor]:
        r"""Computes the KNN edges of a mini-batch.

        Args:
            x_dict (Dict[str, torch.Tensor]): Node embeddings
                :math:`\mathbf{h}^{(0)}` of each node type.
            batch_dict (Dict[str, torch.Tensor], optional): The subgraph of
                each node, per node type. (default: :obj:`None`)
            seed_time (torch.Tensor, optional): The seed time of each
                subgraph. If :obj:`batch_dict` or :obj:`seed_time` is
                :obj:`None`, every other node is a candidate.
                (default: :obj:`None`)

        Returns:
            (Tuple[str, str, str], torch.Tensor): The edge type and the
            :obj:`edge_index` of shape :obj:`[2, num_edges]`.
        """
        x = x_dict[self.node_type]
        batch = None if batch_dict is None else batch_dict.get(self.node_type)
        edge_index = similar_edges(x, self.k,
                                   node_time=node_seed_time(batch, seed_time),
                                   chunk_size=self.chunk_size)
        return self.edge_type, edge_index

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.node_type!r}, k={self.k})'


class BehavioralEdges(torch.nn.Module):
    r"""The *behavioral similarity* injection template from the `"What Makes
    a Desired Graph for Relational Deep Learning?"
    <https://arxiv.org/abs/2606.08491>`_ paper (Eq. 8).

    Two nodes of :obj:`node_type` are linked if they behave alike, *i.e.*, if
    the mean embeddings of their neighbors along the relation :obj:`via` are
    similar:

    .. math::
        \mathcal{E}_{\text{sim}} = \left\{ (u, v) \;\middle|\; \cos\left(
        \bar{\mathbf{h}}_u, \bar{\mathbf{h}}_v \right) >
        \tau_{\text{sim}} \right\}, \qquad
        \bar{\mathbf{h}}_v = \operatorname{mean}_{w \in \mathcal{N}_B(v)}
        \mathbf{h}^{(0)}_w .

    Nodes without neighbors along :obj:`via` take no part. As in
    :class:`KNNEdges`, candidates from newer subgraphs are excluded, and each
    node keeps at most :obj:`max_neighbors` incoming edges, from the most
    similar candidates. The edges are returned under the edge type
    :obj:`(node_type, 'behavioral__<src>__<rel>', node_type)`, with
    :obj:`via = (src, rel, node_type)`.

    Args:
        node_type (str): The node type to connect.
        via (Tuple[str, str, str]): The relation :obj:`(src, rel, node_type)`
            whose neighbors describe a node's behavior.
        threshold (float, optional): The similarity threshold
            :math:`\tau_{\text{sim}}`. (default: :obj:`0.9`)
        max_neighbors (int, optional): The maximum number of incoming edges
            per node. (default: :obj:`10`)
        chunk_size (int, optional): See :func:`similar_edges`.
            (default: :obj:`2048`)

    Example:
        >>> behavioral = BehavioralEdges(
        ...     'drivers', via=('results', 'f2p_driverId', 'drivers'))
        >>> edge_type, edge_index = behavioral(
        ...     x_dict, batch.edge_index_dict, batch.batch_dict,
        ...     batch['drivers'].seed_time)
    """
    def __init__(
        self,
        node_type: NodeType,
        via: EdgeType,
        threshold: float = 0.9,
        max_neighbors: int = 10,
        chunk_size: int = 2048,
    ):
        super().__init__()
        if via[-1] != node_type:
            raise ValueError(f"'via' must point into '{node_type}' "
                             f"(got {via})")
        self.node_type = node_type
        self.via = tuple(via)
        self.threshold = threshold
        self.max_neighbors = max_neighbors
        self.chunk_size = chunk_size

    @property
    def edge_type(self) -> EdgeType:
        r"""The edge type of the added edges."""
        src, rel, _ = self.via
        return (self.node_type, f'behavioral__{src}__{rel}', self.node_type)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        batch_dict: Optional[Dict[NodeType, Tensor]] = None,
        seed_time: Optional[Tensor] = None,
    ) -> Tuple[EdgeType, Tensor]:
        r"""Computes the behavioral edges of a mini-batch.

        Args:
            x_dict (Dict[str, torch.Tensor]): Node embeddings
                :math:`\mathbf{h}^{(0)}` of each node type.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): The
                mini-batch connectivity; edges of :obj:`via` that are missing
                give no edges.
            batch_dict (Dict[str, torch.Tensor], optional): The subgraph of
                each node, per node type. (default: :obj:`None`)
            seed_time (torch.Tensor, optional): The seed time of each
                subgraph. (default: :obj:`None`)

        Returns:
            (Tuple[str, str, str], torch.Tensor): The edge type and the
            :obj:`edge_index` of shape :obj:`[2, num_edges]`.
        """
        x = x_dict[self.node_type]
        edge_index = edge_index_dict.get(self.via)
        if edge_index is None or edge_index.numel() == 0:
            return self.edge_type, x.new_empty((2, 0), dtype=torch.long)
        num_nodes = x.size(0)
        x_src = x_dict[self.via[0]]
        with torch.no_grad():
            mean = scatter(x_src[edge_index[0]], edge_index[1], dim=0,
                           dim_size=num_nodes, reduce='mean')
            degree = scatter(torch.ones_like(edge_index[1]), edge_index[1],
                             dim=0, dim_size=num_nodes, reduce='sum')
        active = (degree > 0).nonzero().view(-1)
        batch = None if batch_dict is None else batch_dict.get(self.node_type)
        node_time = node_seed_time(batch, seed_time)
        out = similar_edges(
            mean[active], self.max_neighbors, threshold=self.threshold,
            node_time=None if node_time is None else node_time[active],
            chunk_size=self.chunk_size)
        return self.edge_type, active[out]

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.node_type!r}, '
                f'via={self.via}, threshold={self.threshold})')


class TwoHopEdges(torch.nn.Module):
    r"""The *two-hop shortcut* injection template from the `"What Makes a
    Desired Graph for Relational Deep Learning?"
    <https://arxiv.org/abs/2606.08491>`_ paper (Eq. 7).

    For two relations :math:`A \xrightarrow{r_1} B \xrightarrow{r_2} C`, adds
    a direct edge for every path through some node :math:`b`:

    .. math::
        \mathcal{E}_{\text{2hop}} = \left\{ (a, c) \;\middle|\; \exists b:
        (a, b) \in \mathcal{E}_{r_1} \wedge (b, c) \in \mathcal{E}_{r_2}
        \right\} .

    The edges are composed from a mini-batch's sampled edges. With temporal
    sampling, both hops then lie in one seed's subgraph, which only holds
    rows up to that seed's time, so no shortcut can leak the future (unlike
    shortcuts composed once on the whole graph, which can be created by
    future rows). Each pair :math:`(a, c)` is added once. The edges are
    returned under the edge type :obj:`(A, '2hop__<r1>__<B>__<r2>', C)`.

    Args:
        first (Tuple[str, str, str]): The first relation :math:`(A, r_1, B)`.
        second (Tuple[str, str, str]): The second relation
            :math:`(B, r_2, C)`.

    Example:
        >>> two_hop = TwoHopEdges(('drivers', 'rev_f2p_driverId', 'results'),
        ...                       ('results', 'f2p_raceId', 'races'))
        >>> edge_type, edge_index = two_hop(batch.edge_index_dict,
        ...                                 batch.num_nodes_dict)
    """
    def __init__(self, first: EdgeType, second: EdgeType):
        super().__init__()
        if first[2] != second[0]:
            raise ValueError(f"The relations {first} and {second} do not "
                             f"form a path")
        self.first = tuple(first)
        self.second = tuple(second)

    @staticmethod
    def instances(edge_types: List[EdgeType]) -> List['TwoHopEdges']:
        r"""One template per pair of relations
        :math:`A \to B \to C` in :obj:`edge_types` with :math:`A \neq C`.
        """
        return [
            TwoHopEdges(first, second) for first in edge_types
            for second in edge_types
            if first[2] == second[0] and first[0] != second[2]
        ]

    @property
    def edge_type(self) -> EdgeType:
        r"""The edge type of the added edges."""
        (src, rel1, mid), (_, rel2, dst) = self.first, self.second
        return (src, f'2hop__{rel1}__{mid}__{rel2}', dst)

    def forward(
        self,
        edge_index_dict: Dict[EdgeType, Tensor],
        num_nodes_dict: Dict[NodeType, int],
    ) -> Tuple[EdgeType, Tensor]:
        r"""Computes the shortcut edges of a mini-batch.

        Args:
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): The
                mini-batch connectivity.
            num_nodes_dict (Dict[str, int]): The number of nodes of each node
                type.

        Returns:
            (Tuple[str, str, str], torch.Tensor): The edge type and the
            :obj:`edge_index` of shape :obj:`[2, num_edges]`.
        """
        first = edge_index_dict.get(self.first)
        second = edge_index_dict.get(self.second)
        if first is None or second is None:
            some = first if first is not None else second
            device = None if some is None else some.device
            return self.edge_type, torch.empty((2, 0), dtype=torch.long,
                                               device=device)
        num_mid = num_nodes_dict[self.first[2]]
        # Group the second hop by its middle node:
        order = second[0].argsort(stable=True)
        mid_sorted, dst_sorted = second[0][order], second[1][order]
        count = torch.bincount(mid_sorted, minlength=num_mid)
        ptr = torch.cat([count.new_zeros(1), count.cumsum(0)])
        # Pair every first-hop edge (a, b) with each second hop out of b:
        reps = count[first[1]]
        src = first[0].repeat_interleave(reps)
        start = ptr[first[1]].repeat_interleave(reps)
        offset = torch.arange(src.numel(), device=src.device)
        offset = offset - (reps.cumsum(0) - reps).repeat_interleave(reps)
        dst = dst_sorted[start + offset]
        num_src = num_nodes_dict[self.first[0]]
        num_dst = num_nodes_dict[self.second[2]]
        edge_index = coalesce(torch.stack([src, dst], dim=0),
                              num_nodes=max(num_src, num_dst))
        return self.edge_type, edge_index

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.first}, {self.second})'
