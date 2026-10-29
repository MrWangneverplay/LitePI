import os, pickle
import numpy as np
import torch
import torch.nn.functional as F

SRC = '/mnt/d/王梓杨/项目组/wzy/datasets/cifar-10-batches-py'
OUT = '/root/delphi/run/validation_100'
MODEL = '/root/delphi/python/minionn/pretrained/7_approx/model.npy'
N = 100
SEED = 20261029

os.makedirs(OUT, exist_ok=True)

# ---- plaintext MiniONN (7_approx) built from Delphi's serialized weights ----
w = np.load(MODEL)
print('weights elements:', w.size)

shapes = [
    ('conv', (64, 3, 3, 3)),
    ('conv', (64, 64, 3, 3)),
    ('conv', (64, 64, 3, 3)),
    ('conv', (64, 64, 3, 3)),
    ('conv', (64, 64, 3, 3)),
    ('conv', (64, 64, 1, 1)),
    ('conv', (16, 64, 1, 1)),
    ('fc', (10, 1024)),
]
idx = 0
params = []
for kind, shp in shapes:
    k = int(np.prod(shp))
    W = w[idx:idx + k].reshape(shp)
    idx += k
    b = w[idx:idx + shp[0]]
    idx += shp[0]
    params.append((kind, torch.from_numpy(W.astype(np.float32)), torch.from_numpy(b.astype(np.float32))))
print('consumed:', idx, 'of', w.size)

STRUCT = ['conv', 'conv', 'pool', 'conv', 'conv', 'pool', 'conv', 'conv', 'conv', 'fc']

def poly(x):
    return 0.1992 + 0.5002 * x + 0.1997 * x * x

def forward(img):
    x = torch.from_numpy(img).float().unsqueeze(0)  # 1,3,32,32
    pi = 0
    for s in STRUCT:
        if s == 'conv':
            kind, W, b = params[pi]; pi += 1
            x = F.conv2d(x, W, b, stride=1, padding=(W.shape[2] // 2)) if W.shape[2] > 1 else F.conv2d(x, W, b)
            x = poly(x)
        elif s == 'pool':
            x = F.avg_pool2d(x, 2)
        else:
            kind, W, b = params[pi]; pi += 1
            x = x.reshape(1, -1)
            x = F.linear(x, W, b)
    return x

with open(os.path.join(SRC, 'test_batch'), 'rb') as f:
    b = pickle.load(f, encoding='bytes')
x_all = b[b'data']
y_all = b[b'labels']

rng = np.random.RandomState(SEED)
idx_list = rng.choice(len(x_all), N, replace=False)

correct = []
with torch.no_grad():
    for i, data_idx in enumerate(idx_list):
        img = x_all[data_idx].reshape(3, 32, 32).astype('float64') / 255.0
        np.save(os.path.join(OUT, f'image_{i}.npy'), img.flatten())
        logits = forward(img)
        pred = int(torch.argmax(logits, dim=1).item())
        correct.append(1 if pred == int(y_all[data_idx]) else 0)

np.save(os.path.join(OUT, 'classes.npy'), np.array([int(y_all[j]) for j in idx_list]).astype('int64'))
np.save(os.path.join(OUT, 'plaintext.npy'), np.array(correct).astype('int64'))
print('plaintext accuracy: %.2f%% (%d/%d)' % (100.0 * sum(correct) / N, sum(correct), N))
