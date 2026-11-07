from typing import List, Optional, Tuple

import torch

from torch_geometric.data import HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import EdgeType
from torch_geometric.utils import sort_edge_index


@functional_transform('temporal_continuity')
class TemporalContinuity(BaseTransform):
    r"""Adds the *temporal continuity* injection template from the `"What
    Makes a Desired Graph for Relational Deep Learning?"
    <https://arxiv.org/abs/2606.08491>`_ paper (Eq. 9)
    (functional name: :obj:`temporal_continuity`).

    For a foreign-key edge type :obj:`(record, rel, entity)`, the records
    pointing to the same entity are sorted by time (ties by node index),
    :math:`[v_1, \ldots, v_m]`, and each record is linked to the
    :obj:`num_previous` records before it:

    .. math::
        \mathcal{E}_{\text{time}} = \{ (v_{t-j}, v_t) \mid
        1 \le j \le \text{num\_previous},\ j < t \le m \} .

    With :obj:`num_previous=1` this is the paper's chain of consecutive
    records. The new edges are stored under the edge type
    :obj:`(record, 'next__' + rel, record)`. They always point from an older
    record to a newer one (or one of the same time), so temporal neighbor
    sampling, which drops nodes newer than the seed, never reaches the future
    through them. Records without an entity get no edges.

    Args:
        edge_types (List[Tuple[str, str, str]], optional): The foreign-key
            edge types :obj:`(record, rel, entity)` that define the chains. If
            :obj:`None`, every edge type whose source has the attribute
            :obj:`time_attr` and whose target does not, *i.e.*, records of a
            lasting entity. Records of an entity that has a time itself
            (*e.g.*, the results of one race) usually all share its time, so
            ordering them by time is not meaningful. (default: :obj:`None`)
        num_previous (int, optional): The number of earlier records each
            record is linked to. (default: :obj:`1`)
        time_attr (str, optional): The name of the node time attribute.
            (default: :obj:`"time"`)

    Example:
        >>> # Chain each driver's race results in time order:
        >>> transform = TemporalContinuity([
        ...     ('results', 'f2p_driverId', 'drivers'),
        ... ])
        >>> data = transform(data)
        >>> data['results', 'next__f2p_driverId', 'results'].edge_index
    """
    def __init__(
        self,
        edge_types: Optional[List[EdgeType]] = None,
        num_previous: int = 1,
        time_attr: str = 'time',
    ):
        if num_previous < 1:
            raise ValueError(f"'num_previous' must be at least 1 (got "
                             f"{num_previous})")
        self.edge_types = (None if edge_types is None else
                           [tuple(edge_type) for edge_type in edge_types])
        self.num_previous = num_previous
        self.time_attr = time_attr

    @staticmethod
    def new_edge_type(edge_type: EdgeType) -> Tuple[str, str, str]:
        r"""Returns the edge type added for :obj:`edge_type`."""
        src, rel, _ = edge_type
        return (src, f'next__{rel}', src)

    def chain_edge_types(self, data: HeteroData) -> List[EdgeType]:
        r"""The foreign-key edge types chained in :obj:`data`."""
        if self.edge_types is not None:
            return self.edge_types
        return [(src, rel, dst) for src, rel, dst in data.edge_types
                if src != dst and self.time_attr in data[src]
                and self.time_attr not in data[dst]]

    def forward(self, data: HeteroData) -> HeteroData:
        for edge_type in self.chain_edge_types(data):
            src = edge_type[0]
            if self.time_attr not in data[src]:
                raise ValueError(f"Node type '{src}' has no attribute "
                                 f"'{self.time_attr}'")
            record, entity = data[edge_type].edge_index
            time = data[src][self.time_attr][record]
            # Sort by entity, then time, then record index (stable sorts,
            # applied from the least to the most significant key):
            order = record.argsort(stable=True)
            order = order[time[order].argsort(stable=True)]
            order = order[entity[order].argsort(stable=True)]
            record, entity = record[order], entity[order]

            rows, cols = [], []
            for j in range(1, self.num_previous + 1):
                same = entity[j:] == entity[:-j]
                rows.append(record[:-j][same])
                cols.append(record[j:][same])
            edge_index = torch.stack([torch.cat(rows), torch.cat(cols)])
            data[self.new_edge_type(edge_type)].edge_index = sort_edge_index(
                edge_index, num_nodes=data[src].num_nodes)
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.edge_types}, '
                f'num_previous={self.num_previous})')
