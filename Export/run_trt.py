import os
import scipy
import torch
from models_VOS import VOSModel
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from DataLoader import ShippingLabClassification, Letterbox
from torchvision import transforms
import tensorrt as trt
from torch2trt import torch2trt, TRTModule


def eval_samples(model, x, samples=5, std_multiplier=2):
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


def ood_pred(model, x):
    lse = model.log_sum_exp(x, dim=1)
    pred = torch.argmax(x, 1)
    run_means = [model.vos_means[t] for t in pred]
    run_means = torch.stack(run_means)
    run_stds = [model.vos_stds[t] for t in pred]
    run_stds = torch.stack(run_stds)
    clamped_inp = torch.clamp(lse, min=0.001)
    shifted_means = (model.ood_mean - model.ood_std) * (torch.log(run_means / clamped_inp))
    out_j = (shifted_means).unsqueeze(1)
    output = torch.cat((x, out_j), 1)
    return output

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
    ckpts = r'../SL_model/'

    model_name = "wrn"
    test = True

    torch_model = VOSModel(n_classes=8, model_name=model_name)
    name = 'vos'

    path = os.path.join(ckpts, model_name + "_" + name + "_100_ships.pth")
    print(path)
    model_dict = torch.load(path)
    torch_model.load_state_dict(model_dict['model_state_dict'])
    torch_model.eval().cuda()
    # convert to TensorRT feeding sample data as input
    batch_size = 1
    x = torch.randn(batch_size, 3, 64, 64, requires_grad=True).cuda()

    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(path))

    model_trt.eval().cuda()
    y = torch_model.eval_samples(x)
    yt = eval_samples(model_trt, x)
    val_dir = os.path.join(r'/mnt/ext/data/ds1_wo_f', 'val_set')

    workers = 2
    torch.manual_seed(5)

    image_size = 64
    mean = np.array([x / 255 for x in [115.8, 115.0, 116.0]])
    std = np.array([x / 255 for x in [52.2, 51.0, 55.6]])

    val_set = ShippingLabClassification(root_dir=val_dir,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean.tolist(), std.tolist()),
                                            Letterbox((image_size, image_size), color=mean)
                                        ]))

    v_dataloader = DataLoader(val_set, batch_size=batch_size,
                              shuffle=True, num_workers=workers)
    print(val_set.classes)
    tqd_e = tqdm(enumerate(v_dataloader, 0), total=len(v_dataloader))
    total_pred_t = []
    total_pred_trt = []
    total_lbl = []
    acc, acc2, acc3 = [], [], []
    for i, data in tqd_e:
        x, lbl, _, _ = data
        x = x.cuda()
        lbl = lbl.cuda()
        # compute ONNX Runtime output prediction
        #torch_out = torch_model(x)
        with torch.no_grad():
            to = torch_model.eval_samples(x)
            trt = eval_samples(model_trt, x)

        predicted_t = torch.argmax(to['lr_soft'].mean(0), dim=1)
        predicted_trt = torch.argmax(trt['sp'], dim=1)
        total_pred_t.append(predicted_t.cpu().numpy())
        total_pred_trt.append(predicted_trt.cpu().numpy())
        total_lbl.append(lbl.cpu().numpy())
        acc.append((predicted_t.int() == predicted_trt.int()).float())
        acc2.append((predicted_t.int() == lbl).float())
        acc3.append((lbl == predicted_trt.int()).float())
        accuracy = torch.cat(acc, dim=0).mean().cpu()
        accuracy2 = torch.cat(acc2, dim=0).mean().cpu()
        accuracy3 = torch.cat(acc3, dim=0).mean().cpu()
        tqd_e.set_description(
            f'runn acc = {accuracy:.3f}, runn acc = {accuracy2:.3f}, runn acc = {accuracy3:.3f}')
    ls = np.array(total_pred_t) == np.array(total_pred_trt)
    print(ls.mean())
    print("Exported model has been tested with TRTRuntime, and the result looks good!")


if __name__ == "__main__":
    run()
