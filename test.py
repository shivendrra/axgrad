import axgrad
import axgrad.nn as nn

embedding = nn.Embedding(num_embeddings=10, embedding_dim=5)
indices = axgrad.tensor([1, 3, 4, 7], dtype="int32")

output = embedding(indices)
print("Embedding Output:")
print(output)

loss = output.sum()
loss.backward()
print("Gradient w.r.t. weight matrix:")
print(embedding.weight.grad)