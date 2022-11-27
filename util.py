import casadi as ca
from dataclasses import dataclass, field
import numpy.typing as npt
from typing import Generic, TypeVar


@dataclass(frozen=True)
class VarDict:
    var: ca.MX | ca.DM
    axis: int = field(default=0)


@dataclass(frozen=True)
class Var:
    indices: slice | int

    def __get__(self, obj: VarDict, obj_type=None) -> ca.MX | ca.DM:
        dims = len(obj.var.shape)
        assert dims == 2

        key = (slice(None), ) * obj.axis + (self.indices, ) + \
            (slice(None), ) * (dims - obj.axis - 1)

        return obj.var[key]


class SliceMeta:
    def __getitem__(self, key: slice | int):
        return Var(key)


Slice = SliceMeta()


X = TypeVar('X', bound=VarDict)
U = TypeVar('U', bound=VarDict)
P = TypeVar('P', bound=VarDict)


@dataclass(frozen=True)
class OptiProblem(Generic[X, U, P]):
    opti: ca.Opti
    T: ca.MX
    x: X
    u: U
    p: P


@dataclass(frozen=True)
class OptiSol(Generic[X, U]):
    xup: ca.DM
    lam_g: ca.DM
    x: X
    u: U

    @classmethod
    def save(cls, opti: ca.Opti, x: X, u: U):
        return cls(
            xup=opti.value(opti.x),
            lam_g=opti.value(opti.lam_g),
            x=x.__class__(opti.value(x.var), axis=x.axis),
            u=u.__class__(opti.value(u.var), axis=u.axis)
        )

    def load(self, opti: ca.Opti):
        opti.set_initial(opti.x, self.xup)
        opti.set_initial(opti.lam_g, self.lam_g)
