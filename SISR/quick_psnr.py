"""Quick Y-channel PSNR/SSIM sanity check (skimage) for smoke-test images."""
import os, sys, glob
import numpy as np, cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

os.chdir("/scratch/hongse/ising")
HR = "datasets/benchmark/Urban100/HR"


def to_y(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)


def score(sr_dir, scale, n=3):
    files = sorted(glob.glob(os.path.join(sr_dir, "img_*.png")))[:n]
    ps, ss = [], []
    for f in files:
        name = os.path.basename(f)
        sr = cv2.imread(f); hr = cv2.imread(os.path.join(HR, name))
        h = min(sr.shape[0], hr.shape[0]); w = min(sr.shape[1], hr.shape[1])
        y_sr, y_hr = to_y(sr[:h, :w]), to_y(hr[:h, :w])
        ps.append(psnr(y_hr, y_sr, data_range=255))
        ss.append(ssim(y_hr, y_sr, data_range=255))
    return np.mean(ps), np.mean(ss)


def bicubic(scale, n=3):
    lr_dir = f"datasets/benchmark/Urban100/LR_bicubic/X{scale}"
    files = sorted(glob.glob(os.path.join(lr_dir, "img_*.png")))[:n]
    ps, ss = [], []
    for f in files:
        name = os.path.basename(f)
        lr = cv2.imread(f); hr = cv2.imread(os.path.join(HR, name))
        up = cv2.resize(lr, (lr.shape[1]*scale, lr.shape[0]*scale),
                        interpolation=cv2.INTER_CUBIC)
        h = min(up.shape[0], hr.shape[0]); w = min(up.shape[1], hr.shape[1])
        y_sr, y_hr = to_y(up[:h, :w]), to_y(hr[:h, :w])
        ps.append(psnr(y_hr, y_sr, data_range=255))
        ss.append(ssim(y_hr, y_sr, data_range=255))
    return np.mean(ps), np.mean(ss)


for scale in [2, 4]:
    bp, bs = bicubic(scale)
    gp, gs = score(f"results_gpu/Urban100/X{scale}", scale)
    tp, ts = score(f"results_thrml/Urban100/X{scale}", scale)
    print(f"X{scale}  bicubic={bp:.3f}/{bs:.4f}  "
          f"gpu={gp:.3f}/{gs:.4f}  thrml={tp:.3f}/{ts:.4f}")
