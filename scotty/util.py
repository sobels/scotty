from cvxpy.expressions.expression import Expression
from cvxpy.expressions.constants.parameter import Parameter
from dataclasses import dataclass, field
import numpy.typing as npt
from typing import Generic, TypeVar

T = TypeVar("T", Expression, npt.NDArray)


@dataclass(frozen=True)
class VarDict(Generic[T]):
    var: T
    axis: int = field(default=0)

    @classmethod
    def width(cls):
        return max(f.indices.stop if isinstance(f.indices, slice) else f.indices + 1 for _, f in vars(cls).items() if isinstance(f, Var))


@dataclass(frozen=True)
class Var(Generic[T]):
    indices: slice | int

    def __get__(self, obj: VarDict[T], obj_type=None) -> T:
        dims = len(obj.var.shape)

        key = (slice(None), ) * obj.axis + (self.indices, ) + \
            (slice(None), ) * (dims - obj.axis - 1)

        return obj.var[key]


class SliceMeta:
    def __getitem__(self, key: slice | int):
        return Var(key)


Slice = SliceMeta()


X = TypeVar('X', bound=VarDict, covariant=True)
U = TypeVar('U', bound=VarDict, covariant=True)
P = TypeVar('P', bound=VarDict, covariant=True)


@dataclass
class OptiVars(Generic[X, U, P]):
    T: Parameter
    Tinv: Parameter
    x: X
    u: U
    p: P
