"""Chest X-ray analysis: pretrained TorchXRayVision DenseNet-121 (trained on 8 public datasets:
NIH, CheXpert, MIMIC-CXR, PadChest, RSNA, OpenI, Google, Kaggle) + test-time-augmentation
uncertainty + Grad-CAM localisation."""
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchxrayvision as xrv
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None


def model():
    global _model
    if _model is None:
        _model = xrv.models.DenseNet(weights="densenet121-res224-all").to(DEVICE).eval()
    return _model


def preprocess(img):
    """PIL / ndarray -> (1,1,224,224) tensor in xrv's [-1024,1024] range, plus 224px display image."""
    if isinstance(img, Image.Image):
        img = np.array(img.convert("L"))
    elif img.ndim == 3:
        img = np.array(Image.fromarray(img).convert("L"))
    arr = xrv.datasets.normalize(img.astype(np.float32), 255)[None]
    arr = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])(arr)
    disp = (arr[0] + 1024) / 2048
    return torch.from_numpy(arr)[None].float().to(DEVICE), np.clip(disp, 0, 1)


def _labels():
    return [p for p in model().pathologies if p]


@torch.no_grad()
def predict(x, n_tta=8, seed=0):
    """Returns {pathology: (mean_prob, std)}. 0.5 = calibrated operating threshold of the model."""
    m = model()
    g = torch.Generator(device="cpu").manual_seed(seed)
    aug = torchvision.transforms.RandomAffine(degrees=5, translate=(0.04, 0.04), scale=(0.95, 1.05))
    batch = [x[0]]
    for _ in range(n_tta - 1):
        torch.manual_seed(int(torch.randint(0, 10**6, (1,), generator=g)))
        batch.append(aug(x[0]))
    probs = m(torch.stack(batch)).cpu().numpy()  # (n_tta, 18)
    return {p: (float(probs[:, i].mean()), float(probs[:, i].std()))
            for i, p in enumerate(m.pathologies) if p}


def gradcam(x, pathology):
    """Grad-CAM on the last dense block for one pathology. Returns heatmap (224,224) in [0,1]."""
    m = model()
    idx = list(m.pathologies).index(pathology)
    x = x.clone().requires_grad_(False)
    feats = m.features(x)
    feats.retain_grad()
    pooled = F.adaptive_avg_pool2d(F.relu(feats), 1).flatten(1)
    logit = m.classifier(pooled)[0, idx]
    m.zero_grad()
    logit.backward()
    w = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def overlay(disp, cam, alpha=0.45):
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    base = np.stack([disp] * 3, -1)
    return Image.fromarray((255 * ((1 - alpha) * base + alpha * heat)).astype(np.uint8))
