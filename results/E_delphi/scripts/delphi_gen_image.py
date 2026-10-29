import pickle, os
import numpy as np

SRC = '/mnt/d/王梓杨/项目组/wzy/datasets/cifar-10-batches-py'
OUT = '/root/delphi/run'
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(SRC, 'test_batch'), 'rb') as f:
    b = pickle.load(f, encoding='bytes')

x = b[b'data']
y = b[b'labels']

idx = 0
img = x[idx].reshape(3, 32, 32).astype('float64') / 255.0
cls = np.array(y[idx]).flatten().astype('int64')

np.save(os.path.join(OUT, 'image.npy'), img.flatten())
np.save(os.path.join(OUT, 'class.npy'), cls)
print('idx', idx, 'class', cls, 'image', img.shape, img.min(), img.max())
