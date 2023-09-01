import copy
import torch

def lerp_multi(param_ls: list, weights=None):
    if weights == None:
        weights = [1/len(param_ls) for _ in param_ls]
    # print(f"weights: {weights}")

    target_net = dict()
    for k in param_ls[0]:
        # print(k)

        if 'running' in k:  # running_mean, running_var  requires_grad=True
            fs = torch.zeros(param_ls[0][k].size(), requires_grad=False).cuda()
            for idx, net in enumerate(param_ls):
                fs = fs + net[k].data * weights[idx].data
        else:
            fs = torch.zeros(param_ls[0][k].size(), requires_grad=True).cuda()
            for idx, net in enumerate(param_ls):
                fs = fs + net[k] * weights[idx]
        target_net[k] = fs
    return target_net
