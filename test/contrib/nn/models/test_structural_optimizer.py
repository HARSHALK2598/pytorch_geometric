import pytest
import torch

from torch_geometric.contrib.nn.models import (
    KNNEdges,
    StructuralOptimizer,
    TwoHopEdges,
)
from torch_geometric.nn import (
    MLP,
    GATConv,
    GraphConv,
    HeteroConv,
    LayerNorm,
    SAGEConv,
)
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.testing import onlyOnline, withCUDA, withPackage

NODE_TYPES = ['users', 'items', 'tags']
EDGE_TYPES = [
    ('users', 'buys', 'items'),
    ('items', 'rev_buys', 'users'),
    ('items', 'has', 'tags'),
    ('tags', 'rev_has', 'items'),
]


def make_inputs(num_users=6, num_items=5, channels=8):
    torch.manual_seed(12345)
    x_dict = {
        'users': torch.randn(num_users, channels),
        'items': torch.randn(num_items, channels),
        'tags': torch.randn(2, channels),
    }
    buys = torch.tensor([[0, 1, 2, 3, 4, 5, 0], [0, 1, 1, 2, 3, 4, 4]])
    has = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]])
    edge_index_dict = {
        EDGE_TYPES[0]: buys,
        EDGE_TYPES[1]: buys.flip(0),
        EDGE_TYPES[2]: has,
        EDGE_TYPES[3]: has.flip(0),
    }
    return x_dict, edge_index_dict


class PlainGNN(torch.nn.Module):
    # The plain heterogeneous GNN: SAGEConv per relation, LayerNorm, ReLU,
    # MLP head, built in the same order as StructuralOptimizer.
    def __init__(self, channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            HeteroConv({
                et: SAGEConv((channels, channels), channels)
                for et in EDGE_TYPES
            }) for _ in range(num_layers)
        ])
        self.norms = torch.nn.ModuleList([
            ModuleDict(
                {nt: LayerNorm(channels, mode='node')
                 for nt in NODE_TYPES}) for _ in range(num_layers)
        ])
        self.head = MLP(channels, out_channels=out_channels, norm='batch_norm',
                        num_layers=1)

    def forward(self, x_dict, edge_index_dict, target_type):
        for conv, norm_dict in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: norm_dict[k](x).relu() for k, x in x_dict.items()}
        return self.head(x_dict[target_type])


def test_structural_optimizer_all_off_is_plain_gnn():
    # The PR's Test 4: with every step off, the model is the plain GNN.
    x_dict, edge_index_dict = make_inputs()
    torch.manual_seed(0)
    model = StructuralOptimizer(NODE_TYPES, EDGE_TYPES, 8, 1)
    torch.manual_seed(0)
    plain = PlainGNN(8, 1, 2)
    assert str(model) == ('StructuralOptimizer(num_layers=2, filter=False, '
                          'templates=0, template_gates=False, vib=False)')
    for train in [True, False]:
        model.train(train)
        plain.train(train)
        out = model(x_dict, edge_index_dict, 'users')
        assert torch.equal(out, plain(x_dict, edge_index_dict, 'users'))
    assert model.structural_loss().item() == 0.0


def full_model(**kwargs):
    torch.manual_seed(0)
    return StructuralOptimizer(
        NODE_TYPES, EDGE_TYPES, 8, 1, gated_tables=['items'],
        templates=[KNNEdges('users', k=2), *TwoHopEdges.instances(EDGE_TYPES)],
        template_gates=True, vib=True, lambda_table=0.1, lambda_template=0.01,
        beta_vib=0.001, gate_kwargs=dict(init_logit=0.0), **kwargs)


def test_structural_optimizer_all_on():
    x_dict, edge_index_dict = make_inputs()
    model = full_model()
    assert model.gate.num_gates == 3  # KNN + 2 two-hop relations
    out = model(x_dict, edge_index_dict, 'users', num_targets=4)
    assert out.size() == (4, 1)

    # The penalty of Eq. 21:
    _, table = model.filter.l0_penalties()
    expected = (0.1 * table + 0.01 * model.gate.l0_penalty() +
                0.001 * model._kl)
    loss = model.structural_loss()
    assert torch.allclose(loss, expected)
    (out.sum() + loss).backward()
    assert model.gate.logit.grad.abs().sum() > 0  # Template switches
    assert model.filter.table_gates['items'].logit.grad is not None
    assert model.vib.mu['users'].weight.grad.abs().sum() > 0

    model.eval()  # Deterministic in evaluation:
    with torch.no_grad():
        assert torch.equal(model(x_dict, edge_index_dict, 'users'),
                           model(x_dict, edge_index_dict, 'users'))


def test_structural_optimizer_closed_templates_are_not_built():
    x_dict, edge_index_dict = make_inputs()
    model = full_model().eval()
    with torch.no_grad():
        model.gate.logit.copy_(torch.tensor([10.0, -10.0, -10.0]))
    built = []
    for template in model.templates:
        template.register_forward_hook(
            lambda module, *_: built.append(module.edge_type))
    model(x_dict, edge_index_dict, 'users')
    assert built == [model.templates[0].edge_type]  # KNN only


def test_structural_optimizer_no_future_through_knn():
    # Two subgraphs: user 0-2 (seed time 9), users 3-5 (seed time 1). With
    # KNN only and table gates fixed open, users of the early subgraph must
    # not depend on the late one.
    x_dict, edge_index_dict = make_inputs()
    edge_index_dict = {et: e[:, :0] for et, e in edge_index_dict.items()}
    model = StructuralOptimizer(NODE_TYPES, EDGE_TYPES, 8, 1,
                                templates=[KNNEdges('users', k=5)]).eval()
    batch_dict = {
        'users': torch.tensor([0, 0, 0, 1, 1, 1]),
        'items': torch.zeros(5, dtype=torch.long),
        'tags': torch.zeros(2, dtype=torch.long),
    }
    seed_time = torch.tensor([9, 1])
    out = model(x_dict, edge_index_dict, 'users', batch_dict, seed_time)
    x_dict['users'][:3] = torch.randn(3, 8)  # Change the late subgraph.
    out2 = model(x_dict, edge_index_dict, 'users', batch_dict, seed_time)
    assert torch.allclose(out[3:], out2[3:])  # The early one is unchanged.
    assert not torch.allclose(out[:3], out2[:3])


def gat_conv(edge_type, channels, injected):
    # Bipartite relations: no self-loops; injected ones without bias.
    return GATConv((channels, channels), channels, add_self_loops=False,
                   bias=not injected)


def graph_conv(edge_type, channels, injected):
    return GraphConv((channels, channels), channels, bias=not injected)


@pytest.mark.parametrize('conv', [gat_conv, graph_conv])
def test_structural_optimizer_other_convolutions(conv):
    # The convolution is any MessagePassing layer for bipartite inputs:
    x_dict, edge_index_dict = make_inputs()
    model = full_model(conv=conv)
    assert all(
        isinstance(c, (GATConv, GraphConv)) for layer in model.convs
        for c in layer.convs.values())
    out = model(x_dict, edge_index_dict, 'users', num_targets=4)
    assert out.size() == (4, 1)
    (out.sum() + model.structural_loss()).backward()
    assert model.gate.logit.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x_dict, edge_index_dict, 'users'),
                           model(x_dict, edge_index_dict, 'users'))


@withCUDA
def test_structural_optimizer_device(device):
    x_dict, edge_index_dict = make_inputs()
    x_dict = {k: v.to(device) for k, v in x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
    batch_dict = {
        'users': torch.tensor([0, 0, 0, 1, 1, 1], device=device),
        'items': torch.tensor([0, 0, 0, 1, 1], device=device),
        'tags': torch.tensor([0, 1], device=device),
    }
    seed_time = torch.tensor([9, 1], device=device)
    model = full_model().to(device)
    out = model(x_dict, edge_index_dict, 'users', batch_dict, seed_time, 4)
    loss = model.structural_loss()
    assert out.device == device and loss.device == device
    (out.sum() + loss).backward()
    assert model.gate.logit.grad.device == device


@onlyOnline
@withPackage('relbench', 'pyg_lib')
def test_structural_optimizer_rel_f1_epoch():
    # The PR's Test 5: one epoch on rel-f1 end to end, every step on.
    import numpy as np
    from relbench import load_dataset

    from torch_geometric.contrib.transforms import TemporalContinuity
    from torch_geometric.contrib.utils import from_relbench
    from torch_geometric.loader import NeighborLoader

    dataset = load_dataset('rel-f1')
    task = dataset.load_task('driver-top3')
    data = TemporalContinuity()(from_relbench(dataset.get_db()))
    train_table = task.get_table('train')
    table = train_table.df
    seed_time = table[train_table.time_col].values.astype(
        'datetime64[s]').astype(np.int64)
    loader = NeighborLoader(
        data, num_neighbors=[16, 16], time_attr='time',
        input_nodes=(task.entity_table,
                     torch.from_numpy(table[task.entity_col].values)),
        input_time=torch.from_numpy(seed_time), batch_size=256, shuffle=True)
    labels = torch.from_numpy(table[task.target_col].values).float()

    channels = 32
    schema = [et for et in data.edge_types if not et[1].startswith('next__')]
    temporal = [et for et in data.edge_types if et[1].startswith('next__')]

    stats = {  # Per-column mean and std, to standardise the features:
        nt: (torch.nanmean(data[nt].x.float(), 0),
             torch.from_numpy(np.nanstd(data[nt].x.float().numpy(), 0)))
        for nt in data.node_types if 'x' in data[nt]
    }

    def features(store, nt):  # A constant feature for tables without one.
        if 'x' not in store:
            return torch.ones(store.num_nodes, 1)
        mean, std = stats[nt]
        return torch.nan_to_num((store.x.float() - mean) / (std + 1e-6))

    encoders = torch.nn.ModuleDict({
        nt:
        torch.nn.Linear(features(data[nt], nt).size(1), channels)
        for nt in data.node_types
    })
    model = StructuralOptimizer(
        data.node_types, schema, channels, 1,
        gated_tables=[t for t in data.node_types
                      if t != 'drivers'], templates=[KNNEdges('drivers', k=5)],
        injected_edge_types=temporal, template_gates=True, vib=True,
        lambda_table=1e-3, lambda_template=1e-3, beta_vib=1e-3)
    optimizer = torch.optim.Adam([*model.parameters(), *encoders.parameters()],
                                 lr=1e-2)
    losses = []
    for batch in loader:
        x_dict = {
            nt: encoders[nt](features(batch[nt], nt))
            for nt in batch.node_types
        }
        n = batch['drivers'].batch_size
        pred = model(x_dict, batch.edge_index_dict, 'drivers',
                     batch.batch_dict, batch['drivers'].seed_time, n)
        y = labels[batch['drivers'].input_id]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred.view(-1), y) + model.structural_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert len(losses) > 1 and all(np.isfinite(losses))
