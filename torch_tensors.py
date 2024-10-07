import torch

# Initialize a tensor with random values
x = torch.randn(3, 3)
print("x:")
print(x)

# Perform basic math operations on the tensor
y = torch.ones(3, 3)
z = x + y
print("z = x + y:")
print(z)

w = torch.matmul(x, z)
print("w = x @ z:")
print(w)
# Element-wise multiplication of tensors
v = torch.tensor([1, 2, 3])
u = torch.tensor([4, 5, 6])
result = v * u
print("Element-wise multiplication:")
print(result)

# Scalar multiplication of a tensor
a = torch.tensor([1, 2, 3])
scalar = 2
result = scalar * a
print("Scalar multiplication:")
print(result)

# Matrix multiplication of tensors
m1 = torch.tensor([[1, 2], [3, 4]])
m2 = torch.tensor([[5, 6], [7, 8]])
result = torch.matmul(m1, m2)
print("Matrix multiplication:")
print(result)
