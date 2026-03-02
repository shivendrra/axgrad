import time
import axgrad
from axgrad import Tensor
from axgrad import zeros, ones, randn, randint, uniform, fill, linspace, arange

RUNS = 100

def bench(label, fn):
  start = time.perf_counter()
  for _ in range(RUNS): fn()
  elapsed = (time.perf_counter() - start) / RUNS * 1000
  print(f"  {label:<28} {elapsed:.4f} ms/run")

print("\n-- Creation ops --------------------------------")
bench("randn(64, 64)",          lambda: randn(64, 64))
bench("zeros(64, 64)",          lambda: zeros(64, 64))
bench("arange(0, 1000, 1.0)",   lambda: arange(0, 1000, 1.0))
bench("linspace(0,1,100,100)",  lambda: linspace(0.0, 0.01, 1.0, 100))

a, b = randn(64, 64), randn(64, 64)
a.requires_grad, b.requires_grad = True, True
s = Tensor([[1.0, -2.0], [3.0, -4.0]], requires_grad=True)

print("\n-- Binary ops --------------------------------")
bench("add  (64x64 + 64x64)",   lambda: a + b)
bench("sub  (64x64 - 64x64)",   lambda: a - b)
bench("mul  (64x64 * 64x64)",   lambda: a * b)
bench("div  (64x64 / 64x64)",   lambda: a / b)
bench("pow  (64x64 ** 2)",      lambda: a ** 2)
bench("add  scalar (a + 3.0)",  lambda: a + 3.0)

print("\n-- Matmul ------------------------------------")
bench("matmul (64x64 @ 64x64)", lambda: a @ b)

print("\n-- Unary ops --------------------------------─")
bench("log",    lambda: a.log())
bench("exp",    lambda: a.exp())
bench("sqrt",   lambda: a.sqrt())
bench("abs",    lambda: a.abs())
bench("sign",   lambda: a.sign())
bench("neg",    lambda: -a)

print("\n-- Activation ops ----------------------------")
bench("relu",     lambda: a.relu())
bench("sigmoid",  lambda: a.sigmoid())
bench("tanh",     lambda: a.tanh())
bench("gelu",     lambda: a.gelu())
bench("silu",     lambda: a.silu())
bench("softplus", lambda: a.softplus())
bench("swish",    lambda: a.swish())
bench("elu",      lambda: a.elu())
bench("leakyrelu",lambda: a.leakyrelu())

print("\n-- Reduction ops ----------------------------─")
bench("sum  (axis=-1)",          lambda: a.sum(axis=-1))
bench("sum  (axis=0)",           lambda: a.sum(axis=0))
bench("mean (axis=-1)",          lambda: a.mean(axis=-1))
bench("var  (axis=-1)",          lambda: a.var(axis=-1))
bench("std  (axis=-1)",          lambda: a.std(axis=-1))
bench("max  (axis=-1)",          lambda: a.max(axis=-1))
bench("min  (axis=-1)",          lambda: a.min(axis=-1))

print("\n-- Shape ops --------------------------------─")
bench("transpose",               lambda: a.transpose())
bench("flatten",                 lambda: a.flatten())
bench("reshape (64x64->4096)",   lambda: a.reshape((4096,)))
bench("squeeze (2x1x2->2x2)",    lambda: Tensor([[[1.0, 2.0]], [[3.0, 4.0]]]).squeeze(1))
bench("unsqueeze (2x2->2x1x2)",  lambda: s.unsqueeze(1))

print("\n-- Autograd (backward) ----------------------─")
def _backward_bench():
  x, y = randn(8, 8), randn(8, 8)
  x.requires_grad, y.requires_grad = True, True
  z = (x @ y).sum()
  z.backward()

bench("matmul -> sum -> backward", _backward_bench)

def _chain_backward():
  x = randn(8, 8)
  x.requires_grad = True
  z = ((x * x) + x).relu().sum()
  z.backward()

bench("chain ops -> backward",     _chain_backward)

print()