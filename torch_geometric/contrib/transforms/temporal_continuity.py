from typing import List, Tuple

from torch_geometric.data import HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import EdgeType


@functional_transform('temporal_continuity')
class TemporalContinuity(BaseTransform):
    r"""Adds temporal-continuity edges, the *temporal* injection template of
    the `"What Makes a Desired Graph for Relational Deep Learning?"
    <https://arxiv.org/abs/2606.08491>`_ paper
    (functional name: :obj:`temporal_continuity`).

    For every given foreign-key edge type :obj:`(record, rel, entity)`, the
    records that point to the same entity are sorted by their timestamp, and
    each record is linked to the next one, :math:`v_{t-1} \to v_t`. The new
    edges are stored under the edge type :obj:`(record, 'next_' + rel,
    record)`. Edges always point from an older to a newer record, so temporal
    neighbour sampling never leaks future information.

    Args:
        edge_types (List[Tuple[str, str, str]]): The foreign-key edge types
            :obj:`(record, rel, entity)` that define which records form a
            chain.
        time_attr (str, optional): The name of the timestamp attribute of the
            record node type. (default: :obj:`"time"`)

    Example:
        >>> # Chain the race results of each driver in time order:
        >>> transform = TemporalContinuity([
        ...     ('results', 'f2p_driverId', 'drivers'),
        ... ])
        >>> data = transform(data)
        >>> data['results', 'next_f2p_driverId', 'results'].edge_index
    """
    def __init__(
        self,
        edge_types: List[EdgeType],
        time_attr: str = 'time',
    ):
        self.edge_types = [tuple(edge_type) for edge_type in edge_types]
        self.time_attr = time_attr

    @staticmethod
    def new_edge_type(edge_type: EdgeType) -> Tuple[str, str, str]:
        r"""Returns the name of the edge type added for :obj:`edge_type`."""
        src, rel, _ = edge_type
        return (src, f'next_{rel}', src)

    def forward(self, data: HeteroData) -> HeteroData:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.edge_types})'
