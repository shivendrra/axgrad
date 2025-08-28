import pytest
from axgrad import Tensor
import axgrad.nn as nn

class TestParameter:
  def test_parameter_creation(self):
    param = nn.Parameter((3, 4))
    assert param.shape == (3, 4)
    assert param.size == 12
    assert param.ndim == 2
    assert param.requires_grad == True
    assert param.is_parameter == True

  def test_parameter_zero_grad(self):
    param = nn.Parameter((2, 3))
    param.grad = Tensor([[1, 2, 3], [4, 5, 6]])
    param.zero_grad()
    assert param.grad is not None

  def test_parameter_clone(self):
    param = nn.Parameter((2, 2))
    param.set_name("test_param")
    cloned = param.clone()
    assert cloned.shape == param.shape
    assert cloned.get_name() == "test_param"

  def test_parameter_detach(self):
    param = nn.Parameter((2, 2))
    detached = param.detach()
    assert detached.requires_grad == False
    assert detached.shape == param.shape

class TestLinear:
  def test_linear_creation(self):
    linear = nn.Linear(10, 5, bias=True)
    assert linear._in == 10
    assert linear._out == 5
    assert linear.weight.shape == (5, 10)
    assert linear.bias.shape == (5,)

  def test_linear_no_bias(self):
    linear = nn.Linear(10, 5, bias=False)
    assert linear.bias is None

  def test_linear_forward(self):
    linear = nn.Linear(3, 2)
    input_tensor = Tensor([[1, 2, 3], [4, 5, 6]])
    output = linear(input_tensor)
    assert output.shape == (2, 2)

  def test_linear_parameters(self):
    linear = nn.Linear(5, 3)
    params = list(linear.parameters())
    assert len(params) == 1
    for module, name, param in params:
      assert isinstance(param, nn.Parameter)

class TestEmbedding:
  def test_embedding_creation(self):
    emb = nn.Embedding(100, 64)
    assert emb.num_embeddings == 100
    assert emb.embedding_dim == 64
    assert emb.weight.shape == (100, 64)

  def test_embedding_forward(self):
    emb = nn.Embedding(10, 4)
    indices = Tensor([1, 2, 3])
    output = emb(indices)
    assert output.shape == (3, 4)

class TestLoss:
  def test_mse_loss(self):
    loss_fn = nn.MSELoss()
    pred = Tensor([1.0, 2.0, 3.0])
    target = Tensor([1.1, 2.1, 2.9])
    loss = loss_fn(pred, target)
    assert loss.ndim == 0

  def test_mse_loss_reduction_sum(self):
    loss_fn = nn.MSELoss(reduction="sum")
    pred = Tensor([1.0, 2.0])
    target = Tensor([1.5, 2.5])
    loss = loss_fn(pred, target)
    assert loss.ndim == 0

  def test_mse_loss_reduction_none(self):
    loss_fn = nn.MSELoss(reduction="none")
    pred = Tensor([1.0, 2.0])
    target = Tensor([1.5, 2.5])
    loss = loss_fn(pred, target)
    assert loss.shape == (2,)

  def test_mae_loss(self):
    loss_fn = nn.MAELoss()
    pred = Tensor([1.0, 2.0, 3.0])
    target = Tensor([1.1, 2.1, 2.9])
    loss = loss_fn(pred, target)
    assert loss.ndim == 0

  def test_cross_entropy_loss(self):
    loss_fn = nn.CrossEntropy()
    pred = Tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])
    target = Tensor([2, 0])
    loss = loss_fn(pred, target)
    assert loss.ndim == 0

  def test_functional_mse(self):
    pred = Tensor([1.0, 2.0])
    target = Tensor([1.5, 2.5])
    loss = nn.mse(pred, target)
    assert loss.ndim == 0

  def test_functional_mae(self):
    pred = Tensor([1.0, 2.0])
    target = Tensor([1.5, 2.5])
    loss = nn.mae(pred, target)
    assert loss.ndim == 0

  def test_functional_cross_entropy(self):
    pred = Tensor([[1.0, 2.0, 3.0]])
    target = Tensor([2])
    loss = nn.cross_entropy(pred, target)
    assert loss.ndim == 0

class TestActivations:
  def test_tanh(self):
    activation = nn.Tanh()
    x = Tensor([0.5, -0.5, 1.0])
    out = activation(x)
    assert out.shape == x.shape

  def test_sigmoid(self):
    activation = nn.Sigmoid()
    x = Tensor([0.5, -0.5, 1.0])
    out = activation(x)
    assert out.shape == x.shape

  def test_relu(self):
    activation = nn.ReLU()
    x = Tensor([0.5, -0.5, 1.0])
    out = activation(x)
    assert out.shape == x.shape

  def test_leaky_relu(self):
    activation = nn.LeakyReLU(eps=0.01)
    x = Tensor([0.5, -0.5, 1.0])
    out = activation(x)
    assert out.shape == x.shape

  def test_elu(self):
    activation = nn.ELU(alpha=1.0)
    x = Tensor([0.5, -0.5, 1.0])
    out = activation(x)
    assert out.shape == x.shape

  def test_silu(self):
    activation = nn.SiLU()
    x = Tensor([0.5, -0.5, 1.0])
    out = activation(x)
    assert out.shape == x.shape

  def test_gelu(self):
    activation = nn.GELU()
    x = Tensor([0.5, -0.5, 1.0])
    out = activation(x)
    assert out.shape == x.shape

  def test_swish(self):
    activation = nn.Swish(beta=1.0)
    x = Tensor([0.5, -0.5, 1.0])
    out = activation(x)
    assert out.shape == x.shape

  def test_softplus(self):
    activation = nn.Softplus()
    x = Tensor([0.5, -0.5, 1.0])
    out = activation(x)
    assert out.shape == x.shape

class TestNormalization:
  def test_layer_norm(self):
    norm = nn.LayerNorm(10)
    x = Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    out = norm(x)
    assert out.shape == x.shape

  def test_layer_norm_no_affine(self):
    norm = nn.LayerNorm(5, elementwise_affine=False)
    assert norm.weight is None
    assert norm.bias is None

  def test_batch_norm_1d(self):
    norm = nn.BatchNorm1d(5)
    x = Tensor([[1, 2, 3, 4, 5]])
    out = norm(x)
    assert out.shape == x.shape

  def test_rms_norm(self):
    norm = nn.RMSNorm(4)
    x = Tensor([[1, 2, 3, 4]])
    out = norm(x)
    assert out.shape == x.shape

  def test_instance_norm_1d(self):
    norm = nn.InstanceNorm1d(3)
    x = Tensor([[1, 2, 3]])
    out = norm(x)
    assert out.shape == x.shape

  def test_local_response_norm(self):
    norm = nn.LocalResponseNorm(3)
    x = Tensor([1, 2, 3])
    out = norm(x)
    assert out.shape == x.shape

  def test_clip(self):
    clip = nn.Clip(2.0)
    x = Tensor([1, 3, -1, 4])
    out = clip(x)
    assert out.shape == x.shape

  def test_clamp(self):
    clamp = nn.Clamp(-1.0, 2.0)
    x = Tensor([1, 3, -2, 4])
    out = clamp(x)
    assert out.shape == x.shape

class TestOptimizer:
  def test_sgd_creation(self):
    linear = nn.Linear(5, 3)
    optimizer = nn.SGD(linear.parameters(), lr=0.01)
    assert optimizer.lr == 0.01
    assert optimizer.momentum == 0.0

  def test_sgd_with_momentum(self):
    linear = nn.Linear(3, 2)
    optimizer = nn.SGD(linear.parameters(), lr=0.01, momentum=0.9)
    assert optimizer.momentum == 0.9

  def test_sgd_zero_grad(self):
    linear = nn.Linear(3, 2)
    optimizer = nn.SGD(linear.parameters(), lr=0.01)
    for _, _, param in linear.parameters():
      param.grad = Tensor([1, 2, 3])
    optimizer.zero_grad()

  def test_sgd_step(self):
    linear = nn.Linear(2, 1)
    optimizer = nn.SGD(linear.parameters(), lr=0.01)
    
    x = Tensor([[1, 2]])
    y = Tensor([[3]])
    
    pred = linear(x)
    loss = nn.mse(pred, y)
    loss.backward()
    
    old_weight = linear.weight.tolist()
    optimizer.step()
    new_weight = linear.weight.tolist()
    
    assert old_weight != new_weight

  def test_sgd_add_param_group(self):
    linear1 = nn.Linear(3, 2)
    linear2 = nn.Linear(2, 1)
    
    optimizer = nn.SGD(linear1.parameters(), lr=0.01)
    optimizer.add_param_group({'params': list(p for _, _, p in linear2.parameters())})
    
    assert len(optimizer.param_groups) == 2

  def test_sgd_lr_methods(self):
    linear = nn.Linear(3, 2)
    optimizer = nn.SGD(linear.parameters(), lr=0.01)
    
    assert optimizer.get_lr() == 0.01
    optimizer.set_lr(0.02)
    assert optimizer.get_lr() == 0.02

class TestModule:
  def test_custom_module(self):
    class SimpleNet(nn.Module):
      def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(5, 1)

      def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

      def inner_repr(self):
        return ""

    net = SimpleNet()
    x = Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    out = net(x)
    assert out.shape == (1, 1)

  def test_module_parameters(self):
    linear = nn.Linear(5, 3)
    params = list(linear.parameters())
    assert len(params) == 2

  def test_module_train_eval(self):
    linear = nn.Linear(5, 3)
    linear.train()
    for _, _, param in linear.parameters():
      assert param.requires_grad == True
    
    linear.eval()
    for _, _, param in linear.parameters():
      assert param.requires_grad == False

  def test_module_zero_grad(self):
    linear = nn.Linear(3, 2)
    for _, _, param in linear.parameters():
      param.grad = Tensor([1, 2, 3])
    
    linear.zero_grad()

  def test_module_n_params(self):
    linear = nn.Linear(5, 3)
    n_params = linear.n_params()
    expected = 5 * 3 + 3
    assert n_params == expected

  def test_module_named_parameters(self):
    linear = nn.Linear(3, 2)
    named_params = list(linear.named_parameters())
    assert len(named_params) == 2
    
    names = [name for name, _ in named_params]
    assert "weight" in names
    assert "bias" in names

  def test_module_state_dict(self):
    linear = nn.Linear(2, 1)
    state_dict = linear.state_dict()
    assert "weight" in state_dict
    assert "bias" in state_dict

class TestIntegration:
  def test_training_loop(self):
    class SimpleNet(nn.Module):
      def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

      def forward(self, x):
        return self.linear(x)

      def inner_repr(self):
        return ""

    net = SimpleNet()
    optimizer = nn.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    x = Tensor([[1, 2], [3, 4]])
    y = Tensor([[3], [7]])

    for epoch in range(5):
      optimizer.zero_grad()
      pred = net(x)
      loss = criterion(pred, y)
      loss.backward()
      optimizer.step()

  def test_activation_chain(self):
    x = Tensor([[-1, 0, 1, 2]])
    
    activations = [nn.Tanh(), nn.Sigmoid(), nn.ReLU(), nn.GELU(), nn.SiLU()]
    
    for activation in activations:
      out = activation(x)
      assert out.shape == x.shape

  def test_normalization_chain(self):
    x = Tensor([[1, 2, 3, 4]])
    norms = [nn.LayerNorm(4), nn.BatchNorm1d(4), nn.RMSNorm(4)]
    for norm in norms:
      out = norm(x)
      assert out.shape == x.shape

  def test_loss_comparison(self):
    pred = Tensor([1.0, 2.0, 3.0])
    target = Tensor([1.1, 2.1, 2.9])

    mse_loss = nn.MSELoss()(pred, target)
    mae_loss = nn.MAELoss()(pred, target)

    assert mse_loss.ndim == 0
    assert mae_loss.ndim == 0

  def test_embedding_to_linear(self):
    vocab_size, embed_dim = 1000, 128
    seq_len, hidden_dim = 10, 64
    
    embedding = nn.Embedding(vocab_size, embed_dim)
    linear = nn.Linear(embed_dim, hidden_dim)
    
    indices = Tensor([1, 5, 10, 20, 50])
    embedded = embedding(indices)
    output = linear(embedded)
    
    assert embedded.shape == (5, embed_dim)
    assert output.shape == (5, hidden_dim)

if __name__ == "__main__":
  pytest.main([__file__, "-v"])