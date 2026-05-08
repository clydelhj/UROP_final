import io
import os
import glob
import random

import h5py
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms.functional as TF


class H5PatchDataset(torch.utils.data.Dataset):
    """Dataset of 256x256 RGB pathology patches stored as JPEG bytes in HDF5 files.

    Domain A = frozen section (source), Domain B = H&E (target).
    Samples are unpaired: A and B are indexed independently each __getitem__.

    load_to_ram=True  (training):   all JPEG bytes loaded upfront into RAM (~30 GB
                                    per DDP process; requires ~120 GB total across
                                    4 processes on a 125 GB server).
    load_to_ram=False (val/test):   index is built upfront; patches read from disk
                                    on demand.  Use this to avoid loading val/test
                                    data into RAM.

    num_workers MUST be 0 in the DataLoader when load_to_ram=True to prevent
    worker forks from each duplicating the 30 GB resident data.
    """

    def __init__(self, input_path, phase='train', augment=False, load_to_ram=True):
        self.augment = augment
        self.load_to_ram = load_to_ram

        path_A = os.path.join(input_path, phase + '_A')
        path_B = os.path.join(input_path, phase + '_B')

        if load_to_ram:
            print(f'[{phase}] Loading patches into RAM from {path_A} ...', flush=True)
            self.patches_A = self._load_to_ram(path_A)
            print(f'[{phase}] Loading patches into RAM from {path_B} ...', flush=True)
            self.patches_B = self._load_to_ram(path_B)
            print(
                f'[{phase}] Ready: {len(self.patches_A):,} frozen | '
                f'{len(self.patches_B):,} H&E patches in RAM.',
                flush=True,
            )
        else:
            print(f'[{phase}] Indexing patches (stream mode) ...', flush=True)
            self.index_A = self._build_index(path_A)
            self.index_B = self._build_index(path_B)
            print(
                f'[{phase}] Indexed: {len(self.index_A):,} frozen | '
                f'{len(self.index_B):,} H&E patches.',
                flush=True,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_to_ram(self, path):
        patches = []
        for fpath in sorted(glob.glob(os.path.join(path, '*.h5'))):
            with h5py.File(fpath, 'r') as f:
                for p in f['patches']:
                    patches.append(bytes(p))
        return patches

    def _build_index(self, path):
        """Returns a list of (filepath, local_patch_index) pairs."""
        index = []
        for fpath in sorted(glob.glob(os.path.join(path, '*.h5'))):
            with h5py.File(fpath, 'r') as f:
                n = len(f['patches'])
            for i in range(n):
                index.append((fpath, i))
        return index

    def _decode(self, jpeg_bytes):
        return Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')

    def _read_from_disk(self, index, idx):
        fpath, local_idx = index[idx]
        with h5py.File(fpath, 'r') as f:
            jpeg_bytes = bytes(f['patches'][local_idx])
        return self._decode(jpeg_bytes)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        if self.load_to_ram:
            return len(self.patches_A)
        return len(self.index_A)

    def __getitem__(self, idx):
        if self.load_to_ram:
            img_A = self._decode(self.patches_A[idx % len(self.patches_A)])
            img_B = self._decode(self.patches_B[random.randrange(len(self.patches_B))])
        else:
            img_A = self._read_from_disk(self.index_A, idx % len(self.index_A))
            img_B = self._read_from_disk(self.index_B, random.randrange(len(self.index_B)))

        if self.augment:
            if random.random() > 0.5:
                img_A = TF.hflip(img_A)
            if random.random() > 0.5:
                img_A = TF.vflip(img_A)
            if random.random() > 0.5:
                img_B = TF.hflip(img_B)
            if random.random() > 0.5:
                img_B = TF.vflip(img_B)

        # uint8 [0,255] -> float32 [-1,1]
        t_A = TF.to_tensor(img_A) * 2.0 - 1.0
        t_B = TF.to_tensor(img_B) * 2.0 - 1.0

        return t_A, t_B
