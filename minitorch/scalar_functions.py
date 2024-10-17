from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to the given scalar-like values"""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the sum of two scalars."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradient of the addition operation."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the natural logarithm of a scalar."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient ofthe logarithm operation."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the product of two scalars."""
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the gradient of the multiplication operation."""
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the inverse of a scalar."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Compute the gradient of the inverse operation."""
        (a,) = ctx.saved_values
        return (d_output * (-1 / (a * a)),)


class Neg(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the negation of a scalar."""
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Compute the gradient of the negation operation."""
        return (operators.neg(d_output),)


class Sigmoid(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the sigmoid of a scalar."""
        sigmoid_val = operators.sigmoid(a)
        ctx.save_for_backward(sigmoid_val)
        return sigmoid_val

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Compute the gradient of the sigmoid operation."""
        (sigmoid_val,) = ctx.saved_values
        return (d_output * sigmoid_val * (1 - sigmoid_val),)


class ReLU(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the ReLU of a scalar."""
        ctx.save_for_backward(a)
        return float(operators.relu(a))  # 确保返回浮点数

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Compute the gradient of the ReLU operation."""
        (a,) = ctx.saved_values
        return (d_output if a > 0 else 0.0,)


class Exp(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the exponential of a scalar."""
        exp_val = operators.exp(a)
        ctx.save_for_backward(exp_val)
        return exp_val

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Compute the gradient of the exponential operation."""
        (exp_val,) = ctx.saved_values
        return (d_output * exp_val,)


class LT(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the less than comparison of two scalars."""
        return float(operators.lt(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the gradient of the less than operation."""
        return 0.0, 0.0


class EQ(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the equality comparison of two scalars."""
        return float(operators.eq(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the gradient of the equality operation."""
        return 0.0, 0.0
