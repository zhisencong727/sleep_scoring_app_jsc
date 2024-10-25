import torch
import numpy as np

# Create tensors from the lists
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

c = torch.tensor([7.0,8.0,9.0])
d = torch.tensor([10.0,11.0,12.0])

# Stack the tensors along a new dimension (dim=0)
result1 = torch.stack((a, b), dim=0)
result2 = torch.stack((c, d), dim=0)

final = torch.stack((result1,result2),dim=0)

print(final)

windowed_mean = torch.mean(final,dim=(1,2),keepdim=False)
windowed_std = torch.std(final,dim=(1,2),keepdim=False)


print(windowed_mean)
print(windowed_mean.shape)
#print(windowed_std)
