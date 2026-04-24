import h5py
import numpy as np
import os
import torch
from data.base_dataset import BaseDataset, get_transform
from PIL import Image

class H5Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--h5_key_A', type=str, default='data', help='dataset key for domain A')
        parser.add_argument('--h5_key_B', type=str, default='data', help='dataset key for domain B')
        return parser

    def _build_index(self, folder, key):
        index = []
        files = sorted(f for f in os.listdir(folder) if f.endswith('.h5'))
        for fname in files:
            path = os.path.join(folder, fname)
            with h5py.File(path, 'r') as f:
                n = len(f[key])
            index.extend([(path, i) for i in range(n)])
        return index

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.key_A = opt.h5_key_A
        self.key_B = opt.h5_key_B
        self.index_A = self._build_index(os.path.join(opt.dataroot, 'A'), self.key_A)
        self.index_B = self._build_index(os.path.join(opt.dataroot, 'B'), self.key_B)
        self.transform = get_transform(opt)

    def __len__(self):
        return max(len(self.index_A), len(self.index_B))

    def __getitem__(self, index):
        path_A, i_A = self.index_A[index % len(self.index_A)]
        path_B, i_B = self.index_B[index % len(self.index_B)]

        with h5py.File(path_A, 'r') as f:
            img_A = f[self.key_A][i_A][:]
        with h5py.File(path_B, 'r') as f:
            img_B = f[self.key_B][i_B][:]

        def to_pil(arr):
            if arr.dtype != np.uint8:
                denom = arr.max() - arr.min()
                if denom > 0:
                    arr = ((arr - arr.min()) / denom * 255).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)
            return Image.fromarray(arr).convert('RGB')

        A = self.transform(to_pil(img_A))
        B = self.transform(to_pil(img_B))
        return {'A': A, 'B': B, 'A_paths': path_A, 'B_paths': path_B}
