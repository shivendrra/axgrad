from ._tensor import tensor, int8, int16, int32, int64, float16, float32, float64, long, double
from ._functions import *
from ._ops import *
from ._utils import *
from .utils import *

random = random(seed=200)
RNG = RNG(seed=300)

__all__ = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'long', 'double']