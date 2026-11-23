import pytest
import torch

from torch_geometric.contrib.nn.models import (
    BehavioralEdges,
    KNNEdges,
    TwoHopEdges,
    similar_edges,
)
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.testing import withPackage


def edge_set(edge_index):
    return set(map(tuple, edge_index.t().tolist()))


# Four nodes: 0 and 1 point almost the same way, 2 is orthogonal to 0 and
# slightly closer to 1, 3 is opposite to 0:
X = torch.tensor([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [-1.0, 0.0]])


def test_similar_edges():
    # Each node receives from its most similar other node (by hand:
    # 0 <- 1, 1 <- 0, 2 <- 1 (cos 0.0995 > 0), 3 <- 2 (cos 0 > -0.995)):
    out = similar_edges(X, num_neighbors=1)
    assert edge_set(out) == {(1, 0), (0, 1), (1, 2), (2, 3)}
    assert out.requires_grad is False

    # Chunks of one node give the same edges:
    assert edge_set(similar_edges(X, 1, chunk_size=1)) == edge_set(out)

    # A threshold keeps only similarities above it (only 0 and 1 are close):
    out = similar_edges(X, num_neighbors=3, threshold=0.9)
    assert edge_set(out) == {(1, 0), (0, 1)}

    # k is capped by the number of other nodes; one node has none:
    assert similar_edges(X, num_neighbors=10).size(1) == 4 * 3
    assert similar_edges(X[:1], num_neighbors=3).size() == (2, 0)


def test_similar_edges_no_future():
    # Nodes 1 and 2 are from a subgraph at time 3, nodes 0 and 3 at time 5.
    # A node may only receive from nodes of time <= its own (by hand):
    # 0 (5) <- 1; 1 (3) <- 2 (0 is newer); 2 (3) <- 1; 3 (5) <- 2.
    node_time = torch.tensor([5, 3, 3, 5])
    out = similar_edges(X, num_neighbors=1, node_time=node_time)
    assert edge_set(out) == {(1, 0), (2, 1), (1, 2), (2, 3)}
    assert (node_time[out[0]] <= node_time[out[1]]).all()

    # A node with no allowed candidate gets no edge:
    node_time = torch.tensor([0, 9, 9, 9])
    out = similar_edges(X[:2], num_neighbors=1, node_time=node_time[:2])
    assert edge_set(out) == {(0, 1)}


def test_knn_edges():
    knn = KNNEdges('drivers', k=1)
    assert str(knn) == "KNNEdges('drivers', k=1)"
    assert knn.edge_type == ('drivers', 'knn', 'drivers')

    x_dict = {'drivers': X}
    edge_type, out = knn(x_dict)
    assert edge_type == ('drivers', 'knn', 'drivers')
    assert edge_set(out) == {(1, 0), (0, 1), (1, 2), (2, 3)}

    # Subgraphs 0 (time 5) and 1 (time 3), as in a temporal mini-batch:
    batch_dict = {'drivers': torch.tensor([0, 1, 1, 0])}
    seed_time = torch.tensor([5, 3])
    _, out = knn(x_dict, batch_dict, seed_time)
    assert edge_set(out) == {(1, 0), (2, 1), (1, 2), (2, 3)}


def test_behavioral_edges():
    # Drivers 0-3 and results 0-4; driver 3 has no results.
    # Mean result embedding per driver (by hand): 0 -> [1, 0],
    # 1 -> [1, 0.05], 2 -> [0, 1]; only drivers 0 and 1 are similar.
    x_dict = {
        'drivers':
        torch.zeros(4, 2),
        'results':
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.1],
                      [0.0, 1.0]]),
    }
    via = ('results', 'f2p_driverId', 'drivers')
    edge_index_dict = {via: torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 2]])}

    behavioral = BehavioralEdges('drivers', via=via, threshold=0.9)
    assert behavioral.edge_type == ('drivers',
                                    'behavioral__results__f2p_driverId',
                                    'drivers')
    edge_type, out = behavioral(x_dict, edge_index_dict)
    assert edge_type == behavioral.edge_type
    assert edge_set(out) == {(1, 0), (0, 1)}

    # Driver 1 is from a newer subgraph, so it may not reach driver 0:
    batch_dict = {'drivers': torch.tensor([0, 1, 0, 0])}
    _, out = behavioral(x_dict, edge_index_dict, batch_dict,
                        torch.tensor([3, 7]))
    assert edge_set(out) == {(0, 1)}

    # Without edges along 'via', no edges:
    _, out = behavioral(x_dict, {})
    assert out.size() == (2, 0)

    with pytest.raises(ValueError, match="must point into 'drivers'"):
        BehavioralEdges('drivers', via=('drivers', 'rev', 'results'))


def test_two_hop_edges():
    first = ('drivers', 'rev_f2p_driverId', 'results')
    second = ('results', 'f2p_raceId', 'races')
    two_hop = TwoHopEdges(first, second)
    assert two_hop.edge_type == ('drivers',
                                 '2hop__rev_f2p_driverId__results__f2p_raceId',
                                 'races')

    # drivers -> results: 0->0, 0->1, 1->2; results -> races: 0->0, 1->0,
    # 2->1; result 3 has a race but no driver. By hand: driver 0 reaches
    # race 0 twice (kept once), driver 1 reaches race 1.
    edge_index_dict = {
        first: torch.tensor([[0, 0, 1], [0, 1, 2]]),
        second: torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]]),
    }
    num_nodes_dict = {'drivers': 2, 'results': 4, 'races': 2}
    edge_type, out = two_hop(edge_index_dict, num_nodes_dict)
    assert edge_type == two_hop.edge_type
    assert out.tolist() == [[0, 1], [0, 1]]

    # A middle node with several second hops fans out:
    edge_index_dict[second] = torch.tensor([[0, 0, 2], [0, 1, 1]])
    _, out = two_hop(edge_index_dict, num_nodes_dict)
    assert edge_set(out) == {(0, 0), (0, 1), (1, 1)}

    # A missing relation gives no edges:
    _, out = two_hop({first: edge_index_dict[first]}, num_nodes_dict)
    assert out.size() == (2, 0)

    with pytest.raises(ValueError, match="do not form a path"):
        TwoHopEdges(first, ('races', 'rev_f2p_raceId', 'results'))


def test_two_hop_instances():
    edge_types = [
        ('drivers', 'rev_f2p_driverId', 'results'),
        ('results', 'f2p_driverId', 'drivers'),
        ('results', 'f2p_raceId', 'races'),
        ('races', 'rev_f2p_raceId', 'results'),
    ]
    # Paths A -> B -> C with A != C (round trips such as
    # drivers -> results -> drivers are left out):
    paths = {(t.first, t.second) for t in TwoHopEdges.instances(edge_types)}
    assert paths == {
        (edge_types[0], edge_types[2]),  # drivers -> results -> races
        (edge_types[3], edge_types[1]),  # races -> results -> drivers
    }


def temporal_graph(num_drivers=30, num_results=300, seed=0):
    gen = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data['drivers'].x = torch.randn(num_drivers, 8, generator=gen)
    data['results'].x = torch.randn(num_results, 8, generator=gen)
    data['races'].x = torch.randn(20, 8, generator=gen)
    data['drivers'].time = torch.zeros(num_drivers, dtype=torch.long)
    data['results'].time = torch.randint(0, 100, (num_results, ),
                                         generator=gen)
    data['races'].time = torch.randint(0, 100, (20, ), generator=gen)
    driver = torch.randint(0, num_drivers, (num_results, ), generator=gen)
    race = torch.randint(0, 20, (num_results, ), generator=gen)
    results = torch.arange(num_results)
    data['results', 'f2p_driverId',
         'drivers'].edge_index = torch.stack([results, driver])
    data['drivers', 'rev_f2p_driverId',
         'results'].edge_index = torch.stack([driver, results])
    data['results', 'f2p_raceId',
         'races'].edge_index = torch.stack([results, race])
    data['races', 'rev_f2p_raceId',
         'results'].edge_index = torch.stack([race, results])
    return data


@withPackage('pyg_lib')
def test_injection_never_reaches_newer_subgraphs():
    # A temporal mini-batch as RelBench builds it: one subgraph per seed,
    # each with the rows up to that seed's time.
    data = temporal_graph()
    seeds = torch.arange(30)
    seed_time = torch.randint(0, 100, (30, ),
                              generator=torch.Generator().manual_seed(1))
    loader = NeighborLoader(data, num_neighbors=[8, 8], time_attr='time',
                            input_nodes=('drivers', seeds),
                            input_time=seed_time, batch_size=30,
                            temporal_strategy='uniform')
    batch = next(iter(loader))
    seed_time = batch['drivers'].seed_time
    x_dict = {key: x for key, x in batch.x_dict.items()}

    def allowed(edge_type, edge_index):
        src, _, dst = edge_type
        t_src = seed_time[batch[src].batch[edge_index[0]]]
        t_dst = seed_time[batch[dst].batch[edge_index[1]]]
        return bool((t_src <= t_dst).all())

    for node_type in ['drivers', 'results']:
        edge_type, out = KNNEdges(node_type, k=5)(x_dict, batch.batch_dict,
                                                  seed_time)
        assert out.size(1) > 0 and allowed(edge_type, out)

    via = ('results', 'f2p_driverId', 'drivers')
    edge_type, out = BehavioralEdges('drivers', via,
                                     threshold=-1.0)(x_dict,
                                                     batch.edge_index_dict,
                                                     batch.batch_dict,
                                                     seed_time)
    assert out.size(1) > 0 and allowed(edge_type, out)

    # Two-hop shortcuts stay inside one subgraph:
    for two_hop in TwoHopEdges.instances(batch.edge_types):
        edge_type, out = two_hop(batch.edge_index_dict, batch.num_nodes_dict)
        src, _, dst = edge_type
        assert torch.equal(batch[src].batch[out[0]], batch[dst].batch[out[1]])
