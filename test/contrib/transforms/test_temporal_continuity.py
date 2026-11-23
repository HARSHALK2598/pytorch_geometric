import pytest
import torch

from torch_geometric.contrib.transforms import TemporalContinuity
from torch_geometric.data import HeteroData


def edge_set(edge_index):
    return set(map(tuple, edge_index.t().tolist()))


def make_data():
    # Results 0-6 with times; results point to drivers (no time) and races
    # (with a time). Result 5 has no driver.
    data = HeteroData()
    data['results'].time = torch.tensor([3, 1, 1, 2, 5, 0, 2])
    data['results'].num_nodes = 7
    data['drivers'].num_nodes = 2
    data['races'].time = torch.tensor([0, 9])
    data['races'].num_nodes = 2
    results = torch.tensor([0, 1, 2, 3, 4, 6])
    driver = torch.tensor([0, 1, 0, 0, 1, 0])
    data['results', 'f2p_driverId',
         'drivers'].edge_index = torch.stack([results, driver])
    data['drivers', 'rev_f2p_driverId',
         'results'].edge_index = torch.stack([driver, results])
    data['results', 'f2p_raceId',
         'races'].edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6],
                                             [0, 0, 1, 1, 1, 0, 0]])
    return data


def test_temporal_continuity():
    transform = TemporalContinuity([('results', 'f2p_driverId', 'drivers')])
    assert str(transform) == ("TemporalContinuity([('results', "
                              "'f2p_driverId', 'drivers')], num_previous=1)")
    data = transform(make_data())

    # By hand: driver 0's results in time order are 2 (t=1), 3 (t=2),
    # 6 (t=2, after 3 by index), 0 (t=3); driver 1's are 1 (t=1), 4 (t=5).
    # Result 5 has no driver and gets no edge.
    edge_type = ('results', 'next__f2p_driverId', 'results')
    out = data[edge_type].edge_index
    assert edge_set(out) == {(2, 3), (3, 6), (6, 0), (1, 4)}
    time = data['results'].time
    assert (time[out[0]] <= time[out[1]]).all()  # Never newer -> older.


def test_temporal_continuity_num_previous():
    edge_type = ('results', 'next__f2p_driverId', 'results')
    data = TemporalContinuity([('results', 'f2p_driverId', 'drivers')],
                              num_previous=2)(make_data())
    # Each result also links to the one two places before it:
    assert edge_set(data[edge_type].edge_index) == {(2, 3), (3, 6), (6, 0),
                                                    (1, 4), (2, 6), (3, 0)}
    # More than a group's size adds every earlier record of the group:
    data = TemporalContinuity([('results', 'f2p_driverId', 'drivers')],
                              num_previous=10)(make_data())
    assert edge_set(data[edge_type].edge_index) == {(2, 3), (3, 6), (6, 0),
                                                    (1, 4), (2, 6), (3, 0),
                                                    (2, 0)}

    with pytest.raises(ValueError, match="at least 1"):
        TemporalContinuity(num_previous=0)


def test_temporal_continuity_default_edge_types():
    # By default: records with a time pointing to an entity without one.
    # results -> races is skipped (races have a time), and so are the
    # reverse edges (drivers have no time):
    transform = TemporalContinuity()
    data = make_data()
    assert transform.chain_edge_types(data) == [('results', 'f2p_driverId',
                                                 'drivers')]
    data = transform(data)
    assert ('results', 'next__f2p_driverId', 'results') in data.edge_types
    assert ('results', 'next__f2p_raceId', 'results') not in data.edge_types


def test_temporal_continuity_requires_time():
    data = make_data()
    with pytest.raises(ValueError, match="has no attribute 'time'"):
        TemporalContinuity([('drivers', 'rev_f2p_driverId', 'results')])(data)
