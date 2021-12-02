import os
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

if __name__ == '__main__':
    output_model_dir='app/src/main/assets/'
    INPUT_SIZE=260 #224
    example = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
    filename='enet_b2_8'
    model=torch.load('../models/affectnet_emotions/'+filename+'.pt').cpu()
    model.eval()
    traced_script_module = torch.jit.trace(model, example)
    #traced_script_module.save(filename+'.pt')
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter(output_model_dir+filename+'.ptl')
