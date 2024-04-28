import torch


params = [torch.rand(1, 3, requires_grad=True)]
param_group = {'params': params}

a = torch.optim.AdamW(params=[torch.randn(1, 1, requires_grad=True)], lr=1e-3)
a.param_groups = []


a.add_param_group(param_group=param_group)


print(params)
print(a.param_groups)
