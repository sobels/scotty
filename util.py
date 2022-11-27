import casadi as ca
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VarDict:
    var: ca.MX
    axis: int = field(default=0)


@dataclass(frozen=True)
class Var:
    indices: slice | int

    def __get__(self, obj: VarDict, obj_type=None) -> ca.MX:
        dims = len(obj.var.shape)
        assert dims == 2

        key = (slice(None), ) * obj.axis + (self.indices, ) + \
            (slice(None), ) * (dims - obj.axis - 1)

        return obj.var[key]


class SliceMeta:
    def __getitem__(self, key: slice | int):
        return Var(key)


Slice = SliceMeta()
