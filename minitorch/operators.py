"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1

# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """Multiplies two numbers.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        Product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Identity function.

    Args:
    ----
        x: Input number.

    Returns:
    -------
        The input number x.

    """
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        Sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negates a number.

    Args:
    ----
        x: Input number.

    Returns:
    -------
        Negative of x.

    """
    return -x


def lt(x: float, y: float) -> float:
    """Checks if x is less than y.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        1.0 if x is less than y, 0.0 otherwise.

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if x is equal to y.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        1.0 if x is equal to y, 0.0 otherwise.

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the maximum of two numbers.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        The larger of x and y.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Checks if two numbers are close to each other.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        1.0 if |x - y| < 1e-2, 0.0 otherwise.

    """
    return math.fabs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Computes the sigmoid function.

    Args:
    ----
        x: Input number.

    Returns:
    -------
        Sigmoid of x.

    Notes:
    -----
        Calculated as:
        f(x) = 1.0 / (1.0 + e^(-x)) if x >= 0
             = e^x / (1.0 + e^x) if x < 0
        for numerical stability.

    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Computes the rectified linear unit (ReLU) function.

    Args:
    ----
        x: Input number.

    Returns:
    -------
        x if x > 0, else 0.

    """
    return x if x > 0 else 0


EPS = 1e-6


def log(x: float) -> float:
    """Computes the natural logarithm.

    Args:
    ----
        x: Input number.

    Returns:
    -------
        Natural logarithm of (x + EPS).

    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Computes the exponential function.

    Args:
    ----
        x: Input number.

    Returns:
    -------
        e^x

    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Computes the gradient of log(x) * d.

    Args:
    ----
        x: Input number.
        d: Gradient from the next layer.

    Returns:
    -------
        Gradient of log(x) * d.

    """
    return d / x


def inv(x: float) -> float:
    """Computes the inverse of x.

    Args:
    ----
        x: Input number.

    Returns:
    -------
        1 / x

    """
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Computes the gradient of inv(x) * d.

    Args:
    ----
        x: Input number.
        d: Gradient from the next layer.

    Returns:
    -------
        Gradient of inv(x) * d.

    """
    return d * (-1 / (x * x))


def relu_back(x: float, d: float) -> float:
    """Computes the gradient of relu(x) * d.

    Args:
    ----
        x: Input number.
        d: Gradient from the next layer.

    Returns:
    -------
        Gradient of relu(x) * d.

    """
    return d if x > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map function.

    Args:
    ----
        fn: Function from one value to one value.

    Returns:
    -------
        A function that takes a list, applies `fn` to each element, and returns a new list.

    """

    def apply(ls: Iterable[float]):
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return apply


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list.

    Args:
    ----
        ls: A list of numbers.

    Returns:
    -------
        A new list with all elements negated.

    """
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipwith (or map2) function.

    Args:
    ----
        fn: Function to combine two values.

    Returns:
    -------
        Function that takes two equally sized lists `ls1` and `ls2`, produces a new list by
        applying fn(x, y) on each pair of elements.

    """

    def apply(ls1: Iterable[float], ls2: Iterable[float]):
        assert len(ls1) == len(ls2)
        size = len(ls1)
        ret = []
        for i in range(size):
            ret.append(fn(ls1[i], ls2[i]))
        return ret

    return apply


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add two lists element-wise.

    Args:
    ----
        ls1: First list of numbers.
        ls2: Second list of numbers.

    Returns:
    -------
        A new list with elements being the sum of corresponding elements from ls1 and ls2.

    """
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order reduce function.

    Args:
    ----
        fn: Function to combine two values.
        start: Start value x_0.

    Returns:
    -------
        Function that takes a list `ls` of elements x_1 ... x_n and computes the reduction
        fn(x_n, fn(x_(n-1), ... fn(x_1, x_0)...)).

    """

    def apply(ls: Iterable[float]) -> float:
        my_list = list(ls).copy()

        if len(my_list) == 0:
            return start
        else:
            current_value = my_list.pop()
            return fn(current_value, apply(my_list))

    return apply


def sum(ls: Iterable[float]) -> float:
    """Calculate the sum of all elements in a list.

    Args:
    ----
        ls: A list of numbers.

    Returns:
    -------
        The sum of all elements in the list.

    """
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in a list.

    Args:
    ----
        ls: A list of numbers.

    Returns:
    -------
        The product of all elements in the list.

    """
    return reduce(mul, 1.0)(ls)
