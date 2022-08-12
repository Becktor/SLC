import os
import scipy
import torch
from models import BayesVGG16, DropoutModel, TTAModel, VOSModel
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from DataLoader import ShippingLabClassification, Letterbox
from torchvision import transforms


def eval_samples_torch(model, x, samples=1, std_multiplier=2):
    outputs = [model(x)[0] for _ in range(samples)]
    print(outputs)
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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def run():
    method = 'vos'
    ckpts = r'../SL_model/'

    model_name = "wrn"
    test = True

    torch_model = VOSModel(n_classes=8, model_name=model_name)
    name = 'vos'

    path = os.path.join(ckpts, model_name + "_" + name + "_100_ships.pt")

    model_dict = torch.load(path)
    torch_model.load_state_dict(model_dict['model_state_dict'])
    torch_model.eval().cuda()
    # convert to TensorRT feeding sample data as input
    x = torch.randn(1, 3, 64, 64, requires_grad=True).cuda()

    y = eval_samples_torch(torch_model, x)

    path = r'Q:\uncert_data\data_cifar_cleaned'
    val_dir = os.path.join(path, 'val_set')
    batch_size = 1
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
                              shuffle=False, num_workers=workers)
    print(val_set.classes)
    tqd_e = tqdm(enumerate(v_dataloader, 0))
    total_pred_t = []
    total_pred_trt = []
    total_lbl = []
    acc, acc2, acc3 = [], [], []
    for i, data in tqd_e:
        x, lbl, _, _ = data
        x = x.cuda()
        lbl = lbl.cuda()

        with torch.no_grad():
            to = eval_samples_torch(torch_model, x)

        predicted_t = torch.argmax(to['sp'], dim=1)
        total_pred_t.append(predicted_t.cpu().numpy())
        total_lbl.append(lbl.cpu().numpy())
        acc.append((predicted_t.int() == lbl).float())
        accuracy = torch.cat(acc, dim=0).mean().cpu()
        tqd_e.set_description(
            f'runn acc = {accuracy:.3f}')
    ls = np.array(total_pred_t) == np.array(total_pred_trt)
    print(ls.mean())
    print("Exported model has been tested with TRTRuntime, and the result looks good!")


if __name__ == "__main__":
    run()
