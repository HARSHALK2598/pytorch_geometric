import math

import pytest
import torch

from torch_geometric.contrib.nn.models import HardConcreteGate


def test_hard_concrete_gate():
    gate = HardConcreteGate(num_gates=4)
    assert str(gate) == ('HardConcreteGate(4, temperature=0.667, '
                         'threshold=0.5)')
    assert gate.logit.tolist() == [2.0] * 4

    with torch.no_grad():
        gate.logit.fill_(-1.0)
    gate.reset_parameters()
    assert gate.logit.tolist() == [2.0] * 4


def test_hard_concrete_gate_training_samples():
    torch.manual_seed(12345)
    gate = HardConcreteGate(num_gates=3)

    g = gate()
    assert g.size() == (3, )

    g = gate(num_samples=10_000)
    assert g.size() == (10_000, 3)
    assert g.min() >= 0 and g.max() <= 1
    # The stretch and clip make exact zeros and ones occur:
    assert (g == 0).any() and (g == 1).any()
    assert ((g > 0) & (g < 1)).any()


@pytest.mark.parametrize('init_logit', [-1.0, 0.0, 2.0])
def test_hard_concrete_gate_prob_open(init_logit):
    torch.manual_seed(12345)
    gate = HardConcreteGate(num_gates=2, init_logit=init_logit)

    # Closed form: P(g > 0) = sigmoid(logit - t * log(-gamma / zeta)):
    expected = 1 / (1 + math.exp(-(init_logit - 2 / 3 * math.log(1 / 11))))
    assert torch.allclose(gate.prob_open(), torch.tensor([expected] * 2))
    assert math.isclose(gate.l0_penalty().item(), 2 * expected, rel_tol=1e-6)

    # ... matches the empirical open rate:
    g = gate(num_samples=100_000)
    empirical = (g > 0).float().mean(dim=0)
    assert torch.allclose(empirical, torch.tensor([expected] * 2), atol=5e-3)


def test_hard_concrete_gate_eval():
    gate = HardConcreteGate(num_gates=4, threshold=0.6)
    with torch.no_grad():  # sigmoid: 0.12, 0.50, 0.59, 0.62
        gate.logit.copy_(torch.tensor([-2.0, 0.0, 0.35, 0.5]))
    gate.eval()

    g = gate()
    assert g.tolist() == [0.0, 0.0, 0.0, 1.0]
    assert torch.equal(gate(), g)  # Deterministic.
    assert gate(num_samples=3).tolist() == [[0.0, 0.0, 0.0, 1.0]] * 3


def test_hard_concrete_gate_gradients():
    torch.manual_seed(12345)
    gate = HardConcreteGate(num_gates=3, init_logit=0.0)

    gate(num_samples=100).sum().backward()
    assert gate.logit.grad is not None
    assert (gate.logit.grad > 0).all()  # A larger logit opens the switch.

    gate.logit.grad = None
    gate.l0_penalty().backward()
    assert (gate.logit.grad > 0).all()

    # Minimizing the penalty closes the switches:
    optimizer = torch.optim.Adam(gate.parameters(), lr=0.1)
    for _ in range(100):
        optimizer.zero_grad()
        gate.l0_penalty().backward()
        optimizer.step()
    gate.eval()
    assert gate().tolist() == [0.0, 0.0, 0.0]


def test_hard_concrete_gate_invalid_arguments():
    with pytest.raises(ValueError, match="'num_gates' must be positive"):
        HardConcreteGate(num_gates=0)
    with pytest.raises(ValueError, match="gamma < 0 < 1 < zeta"):
        HardConcreteGate(gamma=0.1)
    with pytest.raises(ValueError, match="gamma < 0 < 1 < zeta"):
        HardConcreteGate(zeta=0.9)
    with pytest.raises(ValueError, match="'temperature' must be positive"):
        HardConcreteGate(temperature=0.0)
    with pytest.raises(ValueError, match="'threshold' must be in"):
        HardConcreteGate(threshold=1.0)
