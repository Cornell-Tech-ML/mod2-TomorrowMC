from typing import Sequence
from .module import Parameter
from .scalar import Scalar

class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, parameters: Sequence[Parameter]):
        """Initialize the optimizer with a sequence of parameters.

        Args:
        ----
            parameters: A sequence of parameters to optimize.

        """
        self.parameters = parameters

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        """Initialize the SGD optimizer.

        Args:
        ----
            parameters: A sequence of parameters to optimize.
            lr: Learning rate (default: 1.0).

        """
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """Set the gradients of all parameters to None."""
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None

    def step(self) -> None:
        """Perform a single optimization step."""
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.update(Scalar(p.value.data - self.lr * p.value.derivative))
            elif hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.update(p.value - self.lr * p.value.grad)