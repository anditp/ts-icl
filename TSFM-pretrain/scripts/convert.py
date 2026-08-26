import torch
from safetensors.torch import save_file
from collections import OrderedDict

checkpoint = torch.load("/data/parietal/store2/work/apantea/TSFM-pretrain/checkpoints/saving-snail/step-60000.ckpt", map_location="cpu")

state_dict = checkpoint['state_dict']
print(state_dict.keys())

new_sd = OrderedDict()
for k, v in state_dict.items():
    new_sd[k] = v
    #new_sd[k[len("_orig_mod."):]] = v

save_file(new_sd, "model.safetensors")
