from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

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
    right_vals = list(vals)
    right_vals[arg] += epsilon

    left_vals = list(vals)
    left_vals[arg] -= epsilon

    delta = f(*right_vals) - f(*left_vals)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of this variable."""
        pass

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for this variable."""
        pass

    def is_leaf(self) -> bool:
        """Checks if this variable is a leaf in the computation graph."""
        pass

    def is_constant(self) -> bool:
        """Checks if this variable is a constant."""
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables of this variable."""
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """Applies the chain rule to compute gradients."""
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = []
    result = []

    def visit(n: Variable) -> None:
        if n.is_constant():
            return
        if n.unique_id in visited:
            return
        if not n.is_leaf():
            for input in n.history.inputs:
                visit(input)
        visited.append(n.unique_id)
        result.insert(0, n)

    visit(variable)
    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to compute derivatives for the leaf nodes.

    Args:
    ----
        variable: The right-most variable in the computation graph.
        deriv: The derivative of the final output with respect to the `variable`.

    Returns:
    -------
        None. The function updates the derivative values of each leaf node through `accumulate_derivative`.

    """
    result = topological_sort(variable)
    node2driv = {}
    node2driv[variable.unique_id] = deriv
    for n in result:
        if n.is_leaf():
            continue
        if n.unique_id in node2driv.keys():
            deriv = node2driv[n.unique_id]
        deriv_tmp = n.chain_rule(deriv)
        for key, item in deriv_tmp:
            if key.is_leaf():
                key.accumulate_derivative(item)
                continue
            if key.unique_id in node2driv.keys():
                node2driv[key.unique_id] += item
            else:
                node2driv[key.unique_id] = item


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors."""
        return self.saved_values
