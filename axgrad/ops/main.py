import math
from typing import *
from .functionals import relu, gelu, sigmoid, silu, LeakyRelu

def add_tensor(d1, d2) -> list: return [add_tensor(_a, _b) for _a, _b in zip(d1, d2)] if isinstance(d2, list) else d1 + d2
def sub_tensor(d1, d2) -> list: return [sub_tensor(_a, _b) for _a, _b in zip(d1, d2)] if isinstance(d2, list) else d1 - d2
def mul_tensor(d1, d2) -> list: return [mul_tensor(_a, _b) for _a, _b in zip(d1, d2)] if isinstance(d2, list) else d1 * d2
def div_tensor(d1, d2) -> list: return [div_tensor(_a, _b) for _a, _b in zip(d1, d2)] if isinstance(d2, list) else d1 / d2
def pow_tensor(d1, exp) -> list: return [pow_tensor(_a, exp) for _a in d1] if isinstance(d1, list) else math.pow(d1, exp)
def neg_tensor(d1) -> list: return [neg_tensor(_d) for _d in d1] if isinstance(d1, list) else -d1
def sqrt_tensor(d1) -> list: return [sqrt_tensor(_d) for _d in d1] if isinstance(d1, list) else math.sqrt(d1)
def rsqrt_tensor(d1) -> list: return [rsqrt_tensor(_d) for _d in d1] if isinstance(d1, list) else 1.0 / math.sqrt(d1)
def exp_tensor(d1) -> list: return [exp_tensor(_d) for _d in d1] if isinstance(d1, list) else math.exp(d1)
def log_tensor(d1) -> list: return [log_tensor(_d) for _d in d1] if isinstance(d1, list) else math.log(10, d1)
def ln_tensor(d1) -> list: return [ln_tensor(_d) for _d in d1] if isinstance(d1, list) else math.log(d1)
def abs_tensor(d1) -> list: return [abs_tensor(_d) for _d in d1] if isinstance(d1, list) else abs(d1)

def sin_tensor(d1) -> list: return [sin_tensor(_d) for _d in d1] if isinstance(d1, list) else math.sin(d1)
def sinh_tensor(d1) -> list: return [sinh_tensor(_d) for _d in d1] if isinstance(d1, list) else math.sinh(d1)
def cos_tensor(d1) -> list: return [cos_tensor(_d) for _d in d1] if isinstance(d1, list) else math.cos(d1)
def cosh_tensor(d1) -> list: return [cosh_tensor(_d) for _d in d1] if isinstance(d1, list) else math.cosh(d1)
def tan_tensor(d1) -> list: return [tan_tensor(_d) for _d in d1] if isinstance(d1, list) else math.tan(d1)
def tanh_tensor(d1) -> list: return [tanh_tensor(_d) for _d in d1] if isinstance(d1, list) else math.tanh(d1)

def relu_tensor(d1) -> list: return [relu_tensor(_d) for _d in d1] if isinstance(d1, list) else relu(d1)
def gelu_tensor(d1) -> list: return [gelu_tensor(_d) for _d in d1] if isinstance(d1, list) else gelu(d1)
def sigmoid_tensor(d1) -> list: return [sigmoid_tensor(_d) for _d in d1] if isinstance(d1, list) else sigmoid(d1)
def silu_tensor(d1) -> list: return [silu_tensor(_d) for _d in d1] if isinstance(d1, list) else silu(d1)
def leaky_relu_tensor(d1) -> list: return [leaky_relu_tensor(_d) for _d in d1] if isinstance(d1, list) else LeakyRelu(d1)