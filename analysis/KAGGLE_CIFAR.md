# Run the CIFAR ladder on a free GPU (Kaggle)

This machine has CPU-only PyTorch (`2.14.0+cpu`) and no NVIDIA driver, so the
100-200 epoch runs must go to a GPU.  Kaggle gives about 30 GPU-hours per week
free; a 3-conv/64-channel 100-epoch run is roughly 1.5-3 hours on a T4.

## 1. Create the notebook

- kaggle.com -> Create -> Notebook
- Settings -> Accelerator -> GPU T4 x2
- Internet: on (needed for `git clone` and the CIFAR download)

## 2. One cell

```python
!git clone https://github.com/MrWangneverplay/LitePI.git
%cd LitePI
!python analysis/train_deep_conv.py --root . --conv-depth 3 --activation relu \
  --rgb --bn --channels 64 --hidden 256 --epochs 100 --num-seeds 1 \
  --label-smoothing 0.1
```

Kaggle already ships torch/torchvision; the training path does not need
TenSEAL.  If an import is missing, `!pip install -q <package>` in a separate
cell.

## 3. Next runs

```python
# 200 epochs, 3 seeds, conv3
!python analysis/train_deep_conv.py --root . --conv-depth 3 --activation relu \
  --rgb --bn --channels 64 --hidden 256 --epochs 200 --num-seeds 3 \
  --label-smoothing 0.1

# conv4, 200 epochs, 3 seeds
!python analysis/train_deep_conv.py --root . --conv-depth 4 --activation relu \
  --rgb --bn --channels 64 --hidden 256 --epochs 200 --num-seeds 3 \
  --label-smoothing 0.1

# P2 comparison (HE-only route)
!python analysis/train_deep_conv.py --root . --conv-depth 3,4 --activation p2 \
  --rgb --bn --channels 64 --hidden 256 --epochs 200 --num-seeds 3 \
  --label-smoothing 0.1
```

## 4. Download the results

Each run writes
`results/raw_results/E1/cifar10/deep/<tag>/seed<seed>.json`.
Use the notebook file browser, or copy the JSONs to `/kaggle/working` and fetch
them from the Output tab.

## Local quick signal (CPU)

If you want a rough trend before the GPU run:

```powershell
.venv\Scripts\python.exe analysis\train_deep_conv.py --conv-depth 3 `
  --activation relu --rgb --bn --channels 32 --hidden 128 `
  --epochs 20 --num-seeds 1 --train-subset 10000 --workers 4 `
  --label-smoothing 0.1
```

This uses 10k images and 20 epochs; treat it as a trend, not a final number.
