from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides


class MapProto(Protocol):
    """Protocol for map operations on tensors."""

    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Apply the map operation to a tensor.

        Args:
        ----
            x: Input tensor.
            out: Optional output tensor.

        Returns:
        -------
            Result of the map operation.

        """
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Apply a function elementwise to a tensor."""

        def map_fn(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
            return x

        return map_fn

    @staticmethod
    def cmap(fn: Callable[[float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Apply a function elementwise to a tensor, with optional output tensor."""

        def cmap_fn(x: Tensor, y: Tensor) -> Tensor:
            return x

        return cmap_fn

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Apply a binary function elementwise to two tensors."""

        def zip_fn(x: Tensor, y: Tensor) -> Tensor:
            return x

        return zip_fn

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduce a tensor along a dimension using a binary function.

        Args:
        ----
            fn: The binary function to use for reduction.
            start: The initial value for the reduction.

        Returns:
        -------
            A function that performs the reduction operation.

        """

        def reduce_fn(x: Tensor, dim: int) -> Tensor:
            return x

        return reduce_fn

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Multiply two matrices."""
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
        ----
            ops : tensor operations object see `tensor_ops.py`

        Returns:
        -------
            A collection of tensor functions

        """
        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.id_cmap = ops.cmap(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order tensor map function.

        Args:
        ----
            fn: function from float-to-float to apply.

        Returns:
        -------
            new tensor data

        """
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Higher-order tensor zip function.

        Args:
        ----
            fn: function from two floats-to-float to apply

        Returns:
        -------
            new tensor data

        """
        f = tensor_zip(fn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            if a.shape != b.shape:
                c_shape = shape_broadcast(tuple(a.shape), tuple(b.shape))
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """Higher-order tensor reduce function.

        Args:
        ----
            fn: function from two floats-to-float to apply
            start: starting value for reduction

        Returns:
        -------
            new tensor

        """
        f = tensor_reduce(fn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Matrix multiplication of two tensors.

        Args:
        ----
            a: first tensor
            b: second tensor

        Returns:
        -------
            result of matrix multiplication

        """
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Args:
    ----
        fn: Function from float-to-float to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        in_index = np.zeros(len(in_shape), dtype=np.int32)
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, tuple(out_shape), tuple(in_shape), in_index)
            data = in_storage[index_to_position(tuple(in_index), in_strides)]
            map_data = fn(data)
            out[index_to_position(tuple(out_index), out_strides)] = map_data

    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Args:
    ----
        fn: Function mapping two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        a_index = np.zeros(len(a_shape), dtype=np.int32)
        b_index = np.zeros(len(b_shape), dtype=np.int32)
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, tuple(out_shape), tuple(a_shape), a_index)
            broadcast_index(out_index, tuple(out_shape), tuple(b_shape), b_index)
            a_data = a_storage[index_to_position(tuple(a_index), a_strides)]
            b_data = b_storage[index_to_position(tuple(b_index), b_strides)]
            zip_data = fn(a_data, b_data)
            out[index_to_position(tuple(out_index), out_strides)] = zip_data

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Low-level implementation of tensor reduce.

    Args:
    ----
        fn: Reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            o_index = index_to_position(tuple(out_index), out_strides)
            for j in range(a_shape[reduce_dim]):
                a_index = out_index.copy()
                a_index[reduce_dim] = j
                pos_a = index_to_position(tuple(a_index), a_strides)
                v = fn(a_storage[pos_a], out[o_index])
                out[o_index] = v

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
