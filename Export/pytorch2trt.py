import tensorrt as trt
from torch2trt import torch2trt
import os
import torch
from models import BayesVGG16, DropoutModel, TTAModel, VOSModel

method = 'vos'
torch.manual_seed(0)
ckpts = r'../SL_model/'
model_name = "wrn"
test = True

if method == 'dropout':
    torch_model = DropoutModel(n_classes=8, model_name=model_name)
    name = 'dropout'
elif method == 'vos':
    torch_model = VOSModel(n_classes=8, model_name=model_name)
    name = 'vos'

path = os.path.join(ckpts, model_name + "_" + name + "_100_ships.pt")

model_dict = torch.load(path)
torch_model.load_state_dict(model_dict['model_state_dict'])
torch_model.eval().cuda()
# Input to the model
batch_size = 1
x = torch.randn(batch_size, 3, 64, 64).cuda()
y = torch_model(x)

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(torch_model, [x], fp16_mode=True, default_device_type=trt.DeviceType.DLA)

y_trt = model_trt(x)

# check the output against PyTorch
print(torch.max(torch.abs(y[0] - y_trt[0])))
print(model_trt)
pth = path[:-3] + '.pth'
print(pth)
torch.save(model_trt.state_dict(), pth)
print("saved")