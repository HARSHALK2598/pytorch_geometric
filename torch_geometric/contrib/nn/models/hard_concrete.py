from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter


class HardConcreteGate(torch.nn.Module):
    r"""A vector of learnable on/off switches with a differentiable
    :math:`L_0` penalty, from the `"Learning Sparse Neural Networks through
    L0 Regularization" <https://arxiv.org/abs/1712.01312>`_ paper.

    Each switch :math:`g \in [0, 1]` has a learnable logit :math:`\theta`.
    In training mode, a switch is a stretched and clipped *Hard-Concrete*
    sample, so it can be exactly :math:`0` or :math:`1` while remaining
    differentiable with respect to :math:`\theta`:

    .. math::
        s = \sigma\left(\frac{\log u - \log(1 - u) + \theta}{t}\right),
        \quad u \sim \mathcal{U}(0, 1),
        \qquad g = \min\big(1, \max(0, s(\zeta - \gamma) + \gamma)\big).

    The expected number of open switches is available in closed form,

    .. math::
        \mathbb{E}\|g\|_0 = \sum_i \sigma\left(\theta_i - t \log
        \frac{-\gamma}{\zeta}\right),

    and is returned by :meth:`l0_penalty`. In evaluation mode, each switch is
    a hard :math:`g = \mathbb{I}[\sigma(\theta) > \tau]`.

    Used by :class:`~torch_geometric.contrib.nn.models.RelationalFilter`
    (feature and table switches) and
    :class:`~torch_geometric.contrib.nn.conv.GatedHeteroConv` (relation
    switches).

    Args:
        num_gates (int, optional): The number of switches. (default: :obj:`1`)
        init_logit (float, optional): The initial logit :math:`\theta` of
            every switch. Positive values start the switches mostly open.
            (default: :obj:`2.0`)
        temperature (float, optional): The temperature :math:`t > 0`.
            (default: :obj:`2/3`)
        gamma (float, optional): The lower stretch limit :math:`\gamma < 0`.
            (default: :obj:`-0.1`)
        zeta (float, optional): The upper stretch limit :math:`\zeta > 1`.
            (default: :obj:`1.1`)
        threshold (float, optional): The evaluation-mode threshold
            :math:`\tau` on :math:`\sigma(\theta)`. (default: :obj:`0.5`)

    Shapes:
        - **output:** :math:`(G)` or :math:`(S, G)` if :obj:`num_samples=S`
          is given, where :math:`G` is :obj:`num_gates`.

    Example:
        >>> gate = HardConcreteGate(num_gates=4)
        >>> g = gate()               # soft 0..1 samples in training mode
        >>> penalty = gate.l0_penalty()  # differentiable E||g||_0
        >>> gate.eval()
        >>> g = gate()               # hard 0/1 switches
    """
    def __init__(
        self,
        num_gates: int = 1,
        init_logit: float = 2.0,
        temperature: float = 2. / 3.,
        gamma: float = -0.1,
        zeta: float = 1.1,
        threshold: float = 0.5,
    ):
        super().__init__()
        if not gamma < 0. < 1. < zeta:
            raise ValueError(f"Expected 'gamma < 0 < 1 < zeta' "
                             f"(got gamma={gamma}, zeta={zeta})")
        if temperature <= 0.:
            raise ValueError(f"'temperature' must be positive "
                             f"(got {temperature})")

        self.num_gates = num_gates
        self.init_logit = init_logit
        self.temperature = temperature
        self.gamma = gamma
        self.zeta = zeta
        self.threshold = threshold

        self.logit = Parameter(torch.empty(num_gates))
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        torch.nn.init.constant_(self.logit, self.init_logit)

    def forward(self, num_samples: Optional[int] = None) -> Tensor:
        r"""Returns the switch values.

        Args:
            num_samples (int, optional): If set, draws this many independent
                samples in training mode (useful for Monte-Carlo checks).
                (default: :obj:`None`)
        """
        raise NotImplementedError

    def l0_penalty(self) -> Tensor:
        r"""Returns the expected number of open switches
        :math:`\mathbb{E}\|g\|_0` as a differentiable scalar.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_gates}, '
                f'temperature={self.temperature:.3f}, '
                f'threshold={self.threshold})')
