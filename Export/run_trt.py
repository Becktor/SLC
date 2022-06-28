import os
import scipy
import torch
from models import BayesVGG16, DropoutModel, TTAModel, VOSModel
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from DataLoader import ShippingLabClassification, letterbox
from torchvision import transforms
import tensorrt as trt
from torch2trt import torch2trt, TRTModule


def eval_samples_torch(model, x, samples=5, std_multiplier=2):
    outputs = [model(x)[0] for _ in range(samples)]
    output_stack = torch.stack(outputs)
    log_sum_exps = torch.logsumexp(output_stack, dim=2)
    lse_m = log_sum_exps.mean(0)
    lse_std = log_sum_exps.std(0)
    preds = [torch.softmax(output, dim=1) for output in outputs]
    soft_stack = torch.stack(preds)
    means = soft_stack.mean(axis=0)
    stds = soft_stack.std(axis=0)
    softmax_upper = means + (std_multiplier * stds)
    softmax_lower = means - (std_multiplier * stds)
    return {'mean': means, 'stds': stds, 'sp': means, 'sp_u': softmax_upper,
            'sp_l': softmax_lower, 'preds': soft_stack, 'lse': log_sum_exps, 'lse_m': lse_m, 'lse_s': lse_std}


def eval_samples_trt(model, x, samples=5, std_multiplier=2):
    outputs = [model(x)[0] for i in range(samples)]
    output_stack = np.stack(outputs)
    log_sum_exps = scipy.special.logsumexp(output_stack, axis=2)
    lse_m = log_sum_exps.mean(0)
    lse_std = log_sum_exps.std(0)
    preds = [scipy.special.softmax(output, dim=1) for output in outputs]
    soft_stack = np.stack(preds)
    means = soft_stack.mean(axis=0)
    stds = soft_stack.std(axis=0)
    softmax_upper = means + (std_multiplier * stds)
    softmax_lower = means - (std_multiplier * stds)
    return {'mean': means, 'stds': stds, 'sp': means, 'sp_u': softmax_upper,
            'sp_l': softmax_lower, 'preds': soft_stack, 'lse': log_sum_exps, 'lse_m': lse_m, 'lse_s': lse_std}


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def run():
    method = 'vos'
    torch.manual_seed(0)
    #ckpts = r'/mnt/q/git/SLC/ckpts'
    model_name = "resnet18"
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

    path = os.path.join('../ckpts', model_name + "_" + name + "_100.pt")

    model_dict = torch.load(path)
    torch_model.load_state_dict(model_dict['model_state_dict'])
    torch_model.eval().cuda()
    # convert to TensorRT feeding sample data as input
    x = torch.randn(1, 3, 128, 128, requires_grad=True).cuda()

    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(path[:-2]+'pth'))
    y = eval_samples_torch(torch_model, x)
    yt = eval_samples_torch(model_trt, x)
    val_dir = os.path.join(r'/mnt/ext/data/ds1_wo_f', 'val_set')
    image_size = 128
    batch_size = 1
    workers = 2
    torch.manual_seed(1)
    val_set = ShippingLabClassification(root_dir=val_dir,
                                        transform=transforms.Compose([
                                            letterbox((image_size, image_size)),
                                            transforms.ToTensor()
                                        ]))

    v_dataloader = DataLoader(val_set, batch_size=batch_size,
                              shuffle=True, num_workers=workers)
    tqd_e = tqdm(enumerate(v_dataloader, 0))
    total_pred_t = []
    total_pred_trt = []
    total_lbl = []
    acc = []
    for i, data in tqd_e:
        x, lbl, _, _ = data
        x = x.cuda()
        # compute ONNX Runtime output prediction
        #torch_out = torch_model(x)
        to = eval_samples_torch(torch_model, x)
        trt = eval_samples_torch(model_trt, x)

        predicted_t = torch.argmax(to['sp'], dim=1)
        predicted_trt = torch.argmax(trt['sp'], dim=1)
        total_pred_t.append(predicted_t.cpu().numpy())
        total_pred_trt.append(predicted_trt.cpu().numpy())
        total_lbl.append(lbl.cpu().numpy())
        acc.append((predicted_t.int() == predicted_trt.int()).float())
        accuracy = torch.cat(acc, dim=0).mean().cpu()
        tqd_e.set_description(
            f'runn acc = {accuracy:.3f}')

    ls = np.array(total_pred_t) == np.array(total_pred_trt)
    print(ls.mean())
    print("Exported model has been tested with TRTRuntime, and the result looks good!")


if __name__ == "__main__":
    run()
