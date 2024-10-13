from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Tuple, Protocol, List, Optional


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_list: List[Any] = list(vals)
    vals_list[arg] += epsilon
    f_plus = f(*vals_list)
    vals_list[arg] -= 2 * epsilon
    f_minus = f(*vals_list)
    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    history: Optional[Any]

    def accumulate_derivative(self, x: Any) -> None:
        """Add x to the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.
        """
        ...

    @property
    def unique_id(self) -> int:
        """Return a unique identifier for this Variable."""
        ...

    def is_leaf(self) -> bool:
        """Return True if this variable is a leaf in the computation graph."""
        ...

    def is_constant(self) -> bool:
        """Return True if this variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return an iterable of the parent Variables of this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """Implements the chain rule for this variable.
        Returns an iterable of (Variable, derivative) pairs.
        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    order = []

    def dfs(var: Variable) -> None:
        if var in visited or var.is_constant():
            return
        visited.add(var)
        if hasattr(var, "history") and var.history:
            if var.history.last_fn:
                for parent in var.parents:
                    dfs(parent)
        order.append(var)

    dfs(variable)
    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: The derivative of the output with respect to the variable

    """
    sorted_variables = list(topological_sort(variable))
    gradients = {variable: deriv}

    for var in sorted_variables:
        if var.is_leaf():
            var.accumulate_derivative(gradients[var])
        else:
            for parent, grad in var.chain_rule(gradients[var]):
                if parent not in gradients:
                    gradients[parent] = grad
                else:
                    gradients[parent] += grad


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: List[Any] = field(default_factory=list)

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = list(values)

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the values saved for backward pass."""
        return tuple(self.saved_values)
