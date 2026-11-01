import pytest
import torch

from torch_geometric.contrib.nn.models import RelationalFilter


def test_relational_filter():
    filt = RelationalFilter({
        'users': 3,
        'posts': 2,
        'tags': 0
    }, table_types=['posts', 'tags'])
    assert str(filt) == 'RelationalFilter(num_columns=5, num_tables=2)'
    assert list(filt.column_gates.keys()) == ['users', 'posts']
    assert list(filt.table_gates.keys()) == ['posts', 'tags']
    assert filt.column_gates['users'].num_gates == 3

    with torch.no_grad():
        filt.column_gates['users'].logit.fill_(-1.0)
    filt.reset_parameters()
    assert filt.column_gates['users'].logit.tolist() == [2.0] * 3

    # Table gates default to all node types with columns:
    filt = RelationalFilter({'users': 3, 'posts': 2})
    assert list(filt.table_gates.keys()) == ['users', 'posts']


def test_relational_filter_gate_columns():
    torch.manual_seed(12345)
    filt = RelationalFilter({'users': 3}, table_types=[])
    with torch.no_grad():  # Close the second column only:
        filt.column_gates['users'].logit.copy_(torch.tensor([5., -5., 5.]))
    filt.eval()

    x = torch.randn(4, 3, 8)  # [num_nodes, num_columns, channels]
    out = filt.gate_columns('users', x)
    assert out.size() == (4, 3, 8)
    assert torch.equal(out[:, 0], x[:, 0])
    assert torch.equal(out[:, 1], torch.zeros(4, 8))
    assert torch.equal(out[:, 2], x[:, 2])

    x = torch.randn(4, 3)  # [num_nodes, channels]: gates channels instead.
    out = filt.gate_columns('users', x)
    assert torch.equal(out, x * torch.tensor([1., 0., 1.]))

    # Node types without column gates are passed through:
    x = torch.randn(4, 5, 8)
    assert filt.gate_columns('posts', x) is x

    with pytest.raises(ValueError, match="Expected 3 columns"):
        filt.gate_columns('users', torch.randn(4, 2, 8))


def test_relational_filter_gate_columns_training():
    torch.manual_seed(12345)
    filt = RelationalFilter({'users': 3}, table_types=[])

    x = torch.ones(4, 3, 8)
    out = filt.gate_columns('users', x)
    # One sample per column, shared by every node and channel:
    assert torch.equal(out, out[:1, :, :1].expand(4, 3, 8))
    assert out.min() >= 0 and out.max() <= 1


def test_relational_filter_gate_tables():
    filt = RelationalFilter({'users': 3, 'posts': 2}, table_types=['posts'])
    with torch.no_grad():
        filt.table_gates['posts'].logit.fill_(-5.0)
    filt.eval()

    x_dict = {'users': torch.randn(4, 8), 'posts': torch.randn(5, 8)}
    out_dict = filt.gate_tables(x_dict)
    assert out_dict['users'] is x_dict['users']  # No table gate.
    assert torch.equal(out_dict['posts'], torch.zeros(5, 8))  # Closed.

    with torch.no_grad():
        filt.table_gates['posts'].logit.fill_(5.0)
    out_dict = filt.gate_tables(x_dict)
    assert torch.equal(out_dict['posts'], x_dict['posts'])  # Open.


def test_relational_filter_l0_penalties():
    filt = RelationalFilter({'users': 3, 'posts': 2}, table_types=['posts'])
    column_penalty, table_penalty = filt.l0_penalties()

    p = filt.column_gates['users'].prob_open()[0]  # Same for every gate.
    assert torch.allclose(column_penalty, 5 * p)
    assert torch.allclose(table_penalty, p)

    (column_penalty + table_penalty).backward()
    for gate in list(filt.column_gates.values()) + list(
            filt.table_gates.values()):
        assert (gate.logit.grad > 0).all()

    # No gates at all:
    filt = RelationalFilter({'users': 0}, table_types=[])
    column_penalty, table_penalty = filt.l0_penalties()
    assert float(column_penalty) == 0.0 and float(table_penalty) == 0.0
