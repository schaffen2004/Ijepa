from src.models.TimeSeries_transformer import time_series_base, time_series_transformer_predictor
from src.datasets.TimeSeries_dataloader import make_time_series
from src.masks.random import Random_Mask
import torch
import torch.nn as nn
import numpy as np
from logging import getLogger

logger = getLogger()

# ✅ Khởi tạo dataloader
collator = Random_Mask(window_size=20, segment_size=5)
dataset, data_loader, dist_sampler = make_time_series(
    root_path="./data",
    data_file="trading/XAUUSD_M15.csv",
    window_size=20,
    segment_size=5,
    batch_size=32,
    training=True,
    copy_data=False,
    collator=collator
)

# ✅ Khởi tạo mô hình
context_encoder = time_series_base(window_size=20, num_features=6)
predictor = time_series_transformer_predictor(num_points=20)

# ✅ Hàm debug shape
def debug_tensor(name, tensor):
    print(f"{name}: type={type(tensor)}, shape={tensor.shape if hasattr(tensor, 'shape') else 'n/a'}, dtype={getattr(tensor, 'dtype', 'n/a')}")

# ✅ Hàm kiểm tra giá trị NaN
def check_nan(name, tensor):
    if torch.isnan(tensor).any():
        print(f"⚠️ NaN detected in {name}!")
    else:
        print(f"{name} OK (no NaNs)")

# ✅ Duyệt batch
for batch in data_loader:
    window, masks_enc, masks_pred = batch

    # ✅ Debug input
    debug_tensor("window", window)
    debug_tensor("masks_enc", masks_enc)
    debug_tensor("masks_pred", masks_pred)

    # ✅ Context encoder
    context = context_encoder(window, masks=masks_enc)
    debug_tensor("context (output of encoder)", context)

    # ✅ Predictor
    pred = predictor(context, masks_x=masks_enc, masks=masks_pred)
    debug_tensor("pred (output of predictor)", pred)

    # ✅ Target
    target = context_encoder(window, masks=masks_pred)
    debug_tensor("target", target)

    # ✅ Kiểm tra NaN
    check_nan("pred", pred)
    check_nan("target", target)

    # ✅ Loss
    try:
        loss_fn = nn.MSELoss()
        loss = loss_fn(pred, target)
        print(f"✅ Loss: {loss.item()}")
    except Exception as e:
        print(f"❌ Lỗi khi tính loss: {e}")
        logger.error(f"pred shape: {pred.shape}, target shape: {target.shape}")
        logger.error(f"pred min: {pred.min().item()}, max: {pred.max().item()}")
        logger.error(f"target min: {target.min().item()}, max: {target.max().item()}")

    break  # chỉ debug 1 batch