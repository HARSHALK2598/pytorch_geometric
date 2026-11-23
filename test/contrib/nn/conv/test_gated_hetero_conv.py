import pytest
import torch

from torch_geometric.contrib.nn.conv import GatedHeteroConv
from torch_geometric.contrib.nn.models import HardConcreteGate
from torch_geometric.nn import HeteroConv, SAGEConv

BUYS = ('user', 'buys', 'item')
KNN = ('item', 'knn', 'item')


def make_inputs():
    torch.manual_seed(12345)
    x_dict = {'user': torch.randn(3, 8), 'item': torch.randn(4, 8)}
    edge_index_dict = {
        BUYS: torch.tensor([[0, 1, 2, 2], [0, 0, 1, 3]]),
        KNN: torch.tensor([[1, 0, 3], [0, 2, 2]]),  # items 1 and 3: no edge
    }
    return x_dict, edge_index_dict


def make_convs(out_channels=16):
    torch.manual_seed(0)
    return {
        BUYS: SAGEConv((8, 8), out_channels),
        KNN: SAGEConv(8, out_channels, root_weight=False, bias=False),
    }


def set_gate(conv, open_):
    with torch.no_grad():
        conv.gate.logit.fill_(10.0 if open_ else -10.0)


def test_gated_hetero_conv_is_the_gated_sum():
    # The PR's Test 3: with a hand-set switch g, the output is
    # conv_buys + g * conv_knn exactly.
    x_dict, edge_index_dict = make_inputs()
    convs = make_convs()
    conv = GatedHeteroConv(convs, gated_edge_types=[KNN])
    assert str(conv) == 'GatedHeteroConv(num_relations=2, num_gated=1)'

    buys = convs[BUYS]((x_dict['user'], x_dict['item']), edge_index_dict[BUYS])
    knn = convs[KNN](x_dict['item'], edge_index_dict[KNN])
    for g in [0.0, 0.3, 1.0]:
        out = conv(x_dict, edge_index_dict, gate_values=torch.tensor([g]))
        assert torch.allclose(out['item'], buys + g * knn, atol=1e-6)

    # Without a root term or bias, items with no KNN edge are unchanged by
    # the KNN relation, whatever g is (the gate scales messages only):
    out = conv(x_dict, edge_index_dict, gate_values=torch.tensor([0.7]))
    assert torch.allclose(out['item'][[1, 3]], buys[[1, 3]], atol=1e-6)


def test_gated_hetero_conv_eval_mode():
    x_dict, edge_index_dict = make_inputs()
    convs = make_convs()
    conv = GatedHeteroConv(convs, gated_edge_types=[KNN]).eval()
    buys = convs[BUYS]((x_dict['user'], x_dict['item']), edge_index_dict[BUYS])
    knn = convs[KNN](x_dict['item'], edge_index_dict[KNN])

    set_gate(conv, open_=False)  # Closed: the relation contributes nothing.
    assert torch.equal(conv(x_dict, edge_index_dict)['item'], buys)
    assert conv.open_edge_types() == []
    set_gate(conv, open_=True)  # Open: its full message.
    assert torch.allclose(
        conv(x_dict, edge_index_dict)['item'], buys + knn, atol=1e-6)
    assert conv.open_edge_types() == [KNN]


def test_gated_hetero_conv_skips_closed_relations():
    x_dict, edge_index_dict = make_inputs()
    conv = GatedHeteroConv(make_convs(), gated_edge_types=[KNN]).eval()
    set_gate(conv, open_=False)
    calls = []
    conv.convs[KNN].register_forward_hook(lambda *_: calls.append(1))
    conv(x_dict, edge_index_dict)
    assert calls == []  # Not run at all.


def test_gated_hetero_conv_without_gates_is_hetero_conv():
    x_dict, edge_index_dict = make_inputs()
    gated = GatedHeteroConv(make_convs(), gated_edge_types=[])
    plain = HeteroConv(make_convs())
    assert gated.gate is None and gated.sample_gates() is None
    assert gated.l0_penalty().item() == 0.0
    out, expected = gated(x_dict,
                          edge_index_dict), plain(x_dict, edge_index_dict)
    assert out.keys() == expected.keys()
    for key in expected:
        assert torch.equal(out[key], expected[key])


def test_gated_hetero_conv_shared_gate_and_gradients():
    x_dict, edge_index_dict = make_inputs()
    # At logit 0 the sample lies strictly inside (0, 1), where it passes a
    # gradient (a sample clamped to 0 or 1 passes none):
    torch.manual_seed(1)
    gate = HardConcreteGate(1, init_logit=0.0)
    first = GatedHeteroConv(make_convs(out_channels=8), [KNN], gate=gate)
    second = GatedHeteroConv(make_convs(), [KNN], gate=gate)
    assert first.gate is second.gate  # One switch for both layers.

    g = first.sample_gates()  # One sample, passed to both layers:
    assert 0.0 < g.item() < 1.0
    out = second(
        {
            'item': first(x_dict, edge_index_dict, g)['item'],
            'user': x_dict['user']
        }, edge_index_dict, g)
    out['item'].sum().backward()
    # The switch learns from the task (not only from the L0 penalty):
    assert gate.logit.grad is not None and gate.logit.grad.abs().sum() > 0
    assert first.l0_penalty().requires_grad


def test_gated_hetero_conv_errors():
    with pytest.raises(ValueError, match="without a convolution"):
        GatedHeteroConv(make_convs(), [('item', 'other', 'item')])
    with pytest.raises(ValueError, match="duplicates"):
        GatedHeteroConv(make_convs(), [KNN, KNN])
    with pytest.raises(ValueError, match="Expected a gate with 1"):
        GatedHeteroConv(make_convs(), [KNN], gate=HardConcreteGate(2))
