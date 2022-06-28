import os
import scipy
import torch
import ttach as tta
from torch.utils.data import Dataset, DataLoader
from DataLoader import ShippingLabClassification, letterbox
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from models import GluonResnext50, BayesGluonResnext50, DropoutModel, TTAModel, BayesVGG16, VOSModel
import PIL.Image as Image
import sklearn.metrics as metrics
import torch.onnx


class TransformWrapper(object):
    def __init__(self, transform, n=1):
        self.trans = transform
        self.n = n

    def __call__(self, img):
        imgs = [self.trans(img) for _ in range(self.n)]
        return imgs

def run_net(root_dir, ra, epochs=25, name=''):
    val_dir = root_dir
    data = r'Q:\git\SLC\ckpts'
    dirs = r'Q:\uncert_data\cifar\predictions'
    image_size = 128
    model_name = "mobilenet_v3"
    torch.manual_seed(5)

    n_classes = 8
    if name == 'bayes':
        model = BayesVGG16(n_classes=n_classes)
    elif name == 'dropout':
        model = DropoutModel(n_classes=n_classes, model_name=model_name)
    elif name == 'vos':
        model = VOSModel(n_classes=n_classes, model_name=model_name)
    else:
        model = TTAModel(n_classes=n_classes, model_name=model_name)

    path = os.path.join(data, model_name + "_" + name + "_78.pt")


    model_dict = torch.load(os.path.join(root_dir, path))
    clss = model_dict['classes']
    #clss = {v: k for k, v in classes.items()}
    model.load_state_dict(model_dict['model_state_dict'])
    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    idxss = []
    for files in os.listdir(val_dir):
        if files.endswith('.png'):
            idxss.append(files)

    transform = transforms.Compose([
        letterbox((image_size, image_size)),
        transforms.ToTensor()])

    with torch.no_grad():
        tqd_e = tqdm(idxss)
        for data_name in tqd_e:
            path = os.path.join(val_dir, data_name)
            o_img = Image.open(path).convert("RGB")
            img = transform(o_img).unsqueeze(0).cuda()
            obj = model.evaluate_classification(img, samples=10, std_multiplier=2)
            mean = obj["mean"][0].cpu().numpy()
            cls = np.argmax(mean)
            #plot_uncert(model, img, mean, name, obj, clss)
            save_preds(cls, dirs, o_img, clss, data_name,obj, model)


def save_preds(cls, root_dir, o_img, clss, name, obj, model):
    lbl = clss[cls]
    path = os.path.join(root_dir,  lbl)
    fpath = os.path.join(path, name)
    conf_class = model.cdf_class(obj['lse_m'][0], cls)
    if conf_class > 0.3:
        if os.path.exists(path):
            o_img.save(fpath)
        else:
            os.makedirs(path)
            o_img.save(fpath)


def plot_uncert(model, img, mean, name, obj, classes):
    fig, axes = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [3, 1]})
    pr = obj["preds"][0].cpu().numpy()
    ax1, ax2 = axes.ravel()
    ax1.imshow(img[0].cpu().permute(1, 2, 0).numpy())
    n = classes[np.argmax(mean)]
    # ax1.title.set_text(f'pred:  {n}\nlabel: {l}')
    ax1.set_title(f'pred: {n}', loc='left')
    rev = {v: k for k, v in classes.items()}
    ids = [line.replace("_", " ") for line in list(rev.keys())]
    ax2.boxplot(pr, labels=ids, showmeans=True, meanline=True)
    ax2.title.set_text(f'Boxplot of class prediciton')
    ax2.set_ylim([-0.1, 1.1])
    # ax2.legend([n], bbox_to_anchor=(0.5, -.15))
    for label in ax2.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    if name == 'vos':
        ood_m = obj['lse_m'][0].cpu().numpy()
        ood_s = obj['lse_s'][0].cpu().numpy()
        conf = model.cdf(obj['lse_m'][0]) * 100
        conf_class = model.cdf_class(obj['lse_m'][0], np.argmax(mean)) * 100
        fig.suptitle(f'{name}, conf:{conf:.2f}, conf_c: {conf_class:.2f} μ: {ood_m:.2f}±{ood_s:.2f}')
    else:
        fig.suptitle(f'{name}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    path = r'Q:\uncert_data\cifar\val_set\cifar_boats'
    for x in ['vos']:
        run_net(path, False, name=x)
