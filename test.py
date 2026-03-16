import torch
import argparse
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
import os
import time
import re
import torchvision.transforms.functional as TF
from tqdm import tqdm

from utils.test_dataset import dehaze_test_dataset
from models.model_convnext import fusion_net
from cal_parameters import count_parameters

parser = argparse.ArgumentParser(description='Shadow Removal Inference')
parser.add_argument('--test_dir',   type=str, default='')
parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--batch_size', type=int,  default=1)
parser.add_argument('--device',     type=str,  default='cuda:0')
parser.add_argument('--no_tta', action='store_true', help='Disable hflip TTA')
args = parser.parse_args()

# =========================
# Setup
# =========================
os.makedirs(args.output_dir, exist_ok=True)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")
print(f"🔄 TTA: {'disabled' if args.no_tta else 'enabled'}")

# =========================
# Model
# =========================
if not os.path.exists(args.model_path):
    raise FileNotFoundError(f"Model not found: {args.model_path}")

model = fusion_net()
checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint, strict=True)
model = model.to(device).eval()
print(f"✅ Model loaded from: {args.model_path}")
print(f"📦 Params: {count_parameters(model) / 1e6:.3f} M\n")

# =========================
# Data
# =========================
test_dataset = dehaze_test_dataset(args.test_dir)
test_loader  = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,        # Windows requires 0 (no forking support)
    pin_memory=False,     # pin_memory causes issues on Windows with CUDA
)

# =========================
# Inference
# =========================
total_time = 0.0

with torch.no_grad():
    for inp, name in tqdm(test_loader, desc="🌙 Shadow removal", unit="img"):
        inp = inp.to(device)

        t0 = time.perf_counter()

        # Note: autocast disabled — FFC layers require fp32 (cuFFT fp16 only supports power-of-2 sizes)
        out = model(inp)

        if not args.no_tta:
            # hflip TTA — average original + flipped prediction
            out_flip = TF.hflip(model(TF.hflip(inp)))
            out = (out + out_flip) / 2.0

        out = out.clamp(0, 1)
        total_time += time.perf_counter() - t0

        # Parse image number from name and save
        img_num = re.findall(r"\d+", str(name))[0]
        imwrite(out, os.path.join(args.output_dir, f"{img_num}.png"), value_range=(0, 1))

n = len(test_dataset)
print(f"\n✅ Done! {n} images saved to: {args.output_dir}")
print(f"⏱  Runtime per image: {total_time / n:.4f} s")
print(f"⚡ Total time: {total_time:.2f} s")