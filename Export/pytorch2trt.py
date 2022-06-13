import tensorrt as trt
from torch2trt import torch2trt
import os
import torch
from models import BayesVGG16, DropoutModel, TTAModel, VOSModel

method = 'vos'
torch.manual_seed(0)
ckpts = r'/mnt/q/git/SLC/ckpts'
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

path = os.path.join(ckpts, model_name + "_" + name + "_40.pt")

model_dict = torch.load(path)
torch_model.load_state_dict(model_dict['model_state_dict'])
torch_model.eval().cuda()
# Input to the model
batch_size = 1
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True).cuda()
torch_out = torch_model(x)

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(torch_model, [x], fp16_mode=True, default_device_type=trt.DeviceType.DLA)
