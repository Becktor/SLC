import os
import torch
import numpy as np
from torchvision import transforms
from torch2trt import TRTModule


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


def run():
    ckpts = r'../SL_model/'
    model_name = "wrn"
    name = 'vos'

    path = os.path.join(ckpts, model_name + "_" + name + "_100_ships.pt")

    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(path[:-2]+'pth'))
    model_trt.eval().cuda()

    # convert to TensorRT feeding sample data as input
    x = torch.randn(64, 64, 3).numpy()

    image_size = 64
    mean = np.array([x / 255 for x in [115.8, 115.0, 116.0]])
    std = np.array([x / 255 for x in [52.2, 51.0, 55.6]])
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist()),
        Letterbox((image_size, image_size), color=mean)
    ])
    x = trans(x)
    trt = model_trt(x)
    trt_sampled = eval_samples(model_trt, x)

    print(trt)


class Letterbox:
    def __init__(self,  new_shape=(128, 128), color=(114, 114, 114), stride=32, unorm=None, debug=False):
        self.new_shape = new_shape
        self.color = color
        self.stride = stride
        self.unorm = unorm
        self.debug = debug

    def __call__(self, input_img):
        img = input_img#np.array(input_img)
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[1:]  # current shape [height, width]
        if isinstance(self.new_shape, int):
            self.new_shape = (self.new_shape, self.new_shape)

        # Scale ratio (new / old)
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        #if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
        dh, dw = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]  # wh padding

        dh /= 2
        dw /= 2  # divide padding into 2 sides

        if shape[::-1] != new_unpad:  # resize
            img = transforms.Resize(new_unpad)(img)
            #img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        pad = transforms.Pad((left, top, right, bottom), fill=self.color.mean())
        img = pad(img)
        if img.shape[1] != img.shape[2]:
            print('error')

        return img

if __name__ == "__main__":
    run()
