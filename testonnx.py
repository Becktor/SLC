import os
import scipy
import torch
import onnx
from models import BayesVGG16, DropoutModel, TTAModel, VOSModel
import numpy as np
from tqdm import tqdm
import onnxruntime
from torch.utils.data import DataLoader
from DataLoader import ShippingLabClassification, letterbox
from torchvision import transforms


def eval_samples_torch(model, x, samples=10, std_multiplier=2):
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


def eval_samples_onnx(model, x, samples=10, std_multiplier=2):
    def run_m(a, b):
        onnxruntime.set_seed(b)
        return model.run(None, a)[0]

    outputs = [run_m(x, i) for i in range(samples)]
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


    val_dir = os.path.join(r'Q:\uncert_data\ds1_wo_f', 'val_set')
    image_size = 128
    batch_size = 1
    workers = 1
    torch.manual_seed(1)
    val_set = ShippingLabClassification(root_dir=val_dir,
                                        transform=transforms.Compose([
                                            letterbox((image_size, image_size)),
                                            transforms.ToTensor()
                                        ]))

    v_dataloader = DataLoader(val_set, batch_size=batch_size,
                              shuffle=True, num_workers=workers)
    tqd_e = tqdm(enumerate(v_dataloader, 0))

    ort_session = onnxruntime.InferenceSession("sl_fine_cls.onnx", disabled_optimizers=["EliminateDropout"])

    for i, data in tqd_e:
        x, _, _, _ = data
        # compute ONNX Runtime output prediction
        #torch_out = torch_model(x)
        to = eval_samples_torch(torch_model, x)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        oo = eval_samples_onnx(ort_session, ort_inputs)
        #ort_outs = ort_session.run(None, ort_inputs)
        torch_out = to['mean']
        ort_outs = oo['mean']
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs, rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == "__main__":
    run()
