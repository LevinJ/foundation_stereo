import torch

# 1. Define a tensor with requires_grad=True.
# This makes it a leaf node in the computational graph.
x = torch.tensor(2.0, requires_grad=True)

# 2. Perform the forward pass.
# Autograd tracks each operation and builds the graph.
y = x * 2
z = y + 3

# 3. View the autograd trail.
# The `grad_fn` attribute shows which function created the tensor.
print(f"x.grad_fn: {x.grad_fn}")   # Prints None, as it's a leaf tensor.
print(f"y.grad_fn: {y.grad_fn}")   # Prints <MulBackward0 object...>, pointing to the multiplication operation.
print(f"z.grad_fn: {z.grad_fn}")   # Prints <AddBackward0 object...>, pointing to the addition operation.

# 4. Perform the backward pass.
# We backpropagate from the final tensor to compute gradients.
z.backward()

# 5. View the gradient of the leaf tensor.
# The gradient of the final output `z` with respect to the input `x` is 2.
print(f"x.grad: {x.grad}")