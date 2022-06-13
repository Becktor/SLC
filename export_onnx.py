import os

import scipy
import torch
import onnx
from models import BayesVGG16, DropoutModel, TTAModel, VOSModel
import numpy as np
from tqdm import tqdm

method = 'vos'
torch.manual_seed(0)
ckpts = r'Q:\git\SLC\ckpts'
model_name = "mobilenetv3_rw"
test = True

if method == 'bayes':
    torch_model = BayesVGG16(n_classes=8)
    name = 'bayes'
elif method == 'dropout':
    torch_model = DropoutModel(n_classes=8, model_name=model_name)
    name = 'dropout'
elif method == 'vos':
    torch_model = VOSModel(n_classes=8, model_name=model_name)
    name = 'vos'
else:
    torch_model = TTAModel(n_classes=8, model_name=model_name)
    name = 'tta'

path = os.path.join(ckpts, model_name + "_" + name + "_42.pt")

model_dict = torch.load(path)
torch_model.load_state_dict(model_dict['model_state_dict'])
torch_model.eval()
# Input to the model
batch_size = 1
x = torch.randn(batch_size, 3, 128, 128, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,                  # model being run
                  x,                            # model input (or a tuple for multiple inputs)
                  "sl_fine_cls.onnx",           # where to save the model (can be a file or file-like object)
                  export_params=True,           # store the trained parameter weights inside the model file
                  opset_version=10,             # the ONNX version to export the model to
                  do_constant_folding=True,     # whether to execute constant folding for optimization
                  input_names=['input'],        # the model's input names
                  output_names=['pred', 'output'],      # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}})

onnx_model = onnx.load("sl_fine_cls.onnx")
onnx.checker.check_model(onnx_model)
