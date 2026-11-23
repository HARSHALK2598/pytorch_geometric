import torch

from torch_geometric.contrib.nn.models import VariationalBottleneck


def test_variational_bottleneck():
    torch.manual_seed(12345)
    vib = VariationalBottleneck(['users', 'items'], 8, 4)
    assert str(vib) == 'VariationalBottleneck(num_node_types=2, 8, 4)'
    x_dict = {
        'users': torch.randn(5, 8),
        'items': torch.randn(3, 8),
        'tags': torch.randn(2, 8),  # Not compressed.
    }
    z_dict, kl = vib(x_dict)
    assert z_dict['users'].size() == (5, 4)
    assert z_dict['items'].size() == (3, 4)
    assert z_dict['tags'] is x_dict['tags']

    # The penalty is the KL averaged over each type's nodes, summed over
    # types (paper Eq. 21), computed here by hand:
    expected = 0.0
    for node_type in ['users', 'items']:
        mu = vib.mu[node_type](x_dict[node_type])
        logvar = vib.logvar[node_type](x_dict[node_type]).clamp(-10, 10)
        per_node = 0.5 * (mu**2 + logvar.exp() - logvar - 1).sum(-1)
        expected = expected + per_node.mean()
    assert torch.allclose(kl, expected)

    # Evaluation mode returns the mean:
    vib.eval()
    z_dict, _ = vib(x_dict)
    assert torch.equal(z_dict['users'], vib.mu['users'](x_dict['users']))


def test_variational_bottleneck_sampling():
    torch.manual_seed(0)
    vib = VariationalBottleneck(['users'], 8)
    with torch.no_grad():  # sigma = 2 everywhere:
        vib.logvar['users'].weight.zero_()
        vib.logvar['users'].bias.fill_(2 * torch.log(torch.tensor(2.0)))
    x = torch.randn(1, 8).expand(20000, 8)
    z, kl = vib({'users': x})
    mu = vib.mu['users'](x[:1])
    assert torch.allclose(z['users'].mean(0, keepdim=True), mu, atol=0.05)
    assert torch.allclose(z['users'].std(0), torch.full((8, ), 2.0), atol=0.05)
    kl.backward()  # Differentiable in both maps:
    assert vib.mu['users'].weight.grad.abs().sum() > 0
    assert vib.logvar['users'].weight.grad.abs().sum() > 0


def test_variational_bottleneck_kl_is_zero_at_the_prior():
    vib = VariationalBottleneck(['users'], 8)
    for lin in [vib.mu['users'], vib.logvar['users']]:
        with torch.no_grad():
            lin.weight.zero_()
            lin.bias.zero_()
    _, kl = vib({'users': torch.randn(6, 8)})
    assert kl.abs().item() < 1e-7
    _, kl = vib({'users': torch.randn(0, 8)})  # No nodes: no penalty.
    assert kl.item() == 0.0


def test_variational_bottleneck_large_inputs_stay_finite():
    vib = VariationalBottleneck(['results'], 4)
    z, kl = vib({'results': torch.full((3, 4), 1.5e7)})
    assert torch.isfinite(z['results']).all() and torch.isfinite(kl)
