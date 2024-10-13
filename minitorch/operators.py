"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, TypeVar, List


#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
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
    return float(x + y)


def neg(x: float) -> float:
    """Negates a number.

    Args:
    ----
        x: Input number.

    Returns:
    -------
        Negative of x.

    """
    return float(-x)


def lt(x: float, y: float) -> bool:
    """Checks if x is less than y.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        True if x < y, False otherwise.

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if x is equal to y.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        True if x == y, False otherwise.

    """
    return x == y


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


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close to each other.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        True if |x - y| < 1e-2, False otherwise.

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Computes the sigmoid function.

    Args:
    ----
        x: Input number.

    Returns:
    -------
        Sigmoid of x.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Computes the rectified linear unit (ReLU) function.

    Args:
    ----
        x: Input number.

    Returns:
    -------
        max(0, x)

    """
    return float(max(0.0, x))


def log(x: float) -> float:
    """Computes the natural logarithm.

    Args:
    ----
        x: Input number (must be positive).

    Returns:
    -------
        Natural logarithm of x.

    """
    return float(math.log(x))


def exp(x: float) -> float:
    """Computes the exponential function.

    Args:
    ----
        x: Input number.

    Returns:
    -------
        e^x

    """
    return float(math.exp(x))


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
        x: Input number (non-zero).

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
    return -d / (x * x)


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
    return d if x > 0 else 0.0


#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# ## Task 0.3

# Small practice library of elementary higher-order functions.


T = TypeVar("T")
S = TypeVar("S")


def map(f: Callable[[T], S], lst: List[T]) -> List[S]:
    """Apply a function to each element in a list.

    Args:
    ----
        f: A function that takes an element of type T and returns an element of type S.
        lst: A list of elements of type T.

    Returns:
    -------
        A new list with f applied to each element in lst.

    """
    return [f(x) for x in lst]


def zipWith(f: Callable[[T, S], T], lst1: List[T], lst2: List[S]) -> List[T]:
    """Apply a function to pairs of elements from two lists.

    Args:
    ----
        f: A function that takes two arguments of types T and S and returns a value of type T.
        lst1: A list of elements of type T.
        lst2: A list of elements of type S.

    Returns:
    -------
        A new list with f applied to pairs of elements from lst1 and lst2.

    """
    return [f(x, y) for x, y in zip(lst1, lst2)]


def reduce(f: Callable[[T, T], T], lst: List[T], initial: T) -> T:
    """Reduce a list to a single value by repeatedly applying a binary function.

    Args:
    ----
        f: A function that takes two arguments of type T and returns a value of type T.
        lst: A list of elements of type T.
        initial: The initial value of type T.

    Returns:
    -------
        The result of folding f over the list, starting with the initial value.

    """
    result = initial
    for x in lst:
        result = f(result, x)
    return result


def negList(lst: List[float]) -> List[float]:
    """Negate all elements in a list.

    Args:
    ----
        lst: A list of numbers

    Returns:
    -------
        A new list with all elements negated.

    """
    return map(lambda x: -x, lst)


def addLists(lst1: List[float], lst2: List[float]) -> List[float]:
    """Add two lists element-wise.

    Args:
    ----
        lst1: First list of numbers.
        lst2: Second list of numbers.

    Returns:
    -------
        A new list with elements being the sum of corresponding elements from lst1 and lst2.

    """
    return zipWith(lambda x, y: x + y, lst1, lst2)


def sum(lst: List[float]) -> float:
    """Calculate the sum of all elements in a list.

    Args:
    ----
        lst: A list of numbers

    Returns:
    -------
        The sum of all elements in the list.

    """
    return reduce(lambda x, y: x + y, lst, 0.0)


def prod(lst: List[float]) -> float:
    """Calculate the product of all elements in a list.

    Args:
    ----
        lst: A list of numbers

    Returns:
    -------
        The product of all elements in the list.

    """
    return reduce(lambda x, y: x * y, lst, 1.0)
