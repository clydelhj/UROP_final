import glob
import io
import os
import random
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src import utils

log = utils.get_pylogger(__name__)


class _UnpairedPatchDataset(Dataset):
    """Lazy-loading unpaired dataset. Decodes patches on demand from H5 files.

    Each worker process opens its own H5 file handles to avoid multiprocessing conflicts.
    """

    def __init__(
        self,
        source_index: List[Tuple[str, int]],
        target_index: List[Tuple[str, int]],
        h5_key: str,
        normalize: bool,
    ):
        self.source_index = source_index
        self.target_index = target_index
        self.h5_key = h5_key
        self.normalize = normalize
        self._handles = {}  # worker-local H5 file handles, opened on first access

    def __len__(self):
        return max(len(self.source_index), len(self.target_index))

    def _get_patch(self, index: List[Tuple[str, int]], idx: int) -> torch.Tensor:
        fpath, patch_idx = index[idx]
        if fpath not in self._handles:
            self._handles[fpath] = h5py.File(fpath, "r")
        raw = self._handles[fpath][self.h5_key][patch_idx]
        img = Image.open(io.BytesIO(raw.tobytes()))
        arr = np.array(img, dtype=np.float32)
        if self.normalize:
            arr = arr / 127.5 - 1.0
        if arr.ndim == 2:                          # grayscale (H, W) -> (1, H, W)
            arr = arr[np.newaxis]
        else:                                      # RGB (H, W, C) -> (C, H, W)
            arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int):
        src = self._get_patch(self.source_index, idx % len(self.source_index))
        tgt_idx = random.randint(0, len(self.target_index) - 1)
        tgt = self._get_patch(self.target_index, tgt_idx)
        return src, tgt


class LungPatchH5DataModule(LightningDataModule):
    """Unpaired H5 patch datamodule for virtual staining (frozen -> H&E).

    Builds a patch index at setup time (fast) and decodes patches on demand during
    training — no bulk RAM loading. All patches from all slides are seen each epoch.
    Exposes `dims` as (C, H, W). DDP is handled automatically by PyTorch Lightning.
    """

    pass_to_model = True

    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        h5_key: str = "patches",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.dims = self._peek_dims(source_dir)
        self._dataset: Optional[_UnpairedPatchDataset] = None

    # ------------------------------------------------------------------

    def _h5_files(self, directory: str) -> List[str]:
        if os.path.isfile(directory):
            return [directory]
        files = sorted(
            glob.glob(os.path.join(directory, "**", "*.h5"), recursive=True)
            + glob.glob(os.path.join(directory, "**", "*.hdf5"), recursive=True)
        )
        if not files:
            raise FileNotFoundError(f"No H5 files found in {directory}")
        return files

    def _peek_dims(self, directory: str) -> tuple:
        fpath = self._h5_files(directory)[0]
        with h5py.File(fpath, "r") as f:
            raw = f[self.hparams.h5_key][0]
        img = np.array(Image.open(io.BytesIO(raw.tobytes())))
        if img.ndim == 2:
            return (1, img.shape[0], img.shape[1])
        return (img.shape[2], img.shape[0], img.shape[1])

    def _build_index(self, directory: str, label: str) -> List[Tuple[str, int]]:
        files = self._h5_files(directory)
        index = []
        for fpath in tqdm(files, desc=f"Indexing {label}", unit="file"):
            with h5py.File(fpath, "r") as f:
                n = len(f[self.hparams.h5_key])
            index.extend((fpath, i) for i in range(n))
        log.info(f"{label}: {len(index)} patches across {len(files)} files")
        return index

    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None):
        if self._dataset is None:
            source_index = self._build_index(self.hparams.source_dir, "source (frozen)")
            target_index = self._build_index(self.hparams.target_dir, "target (H&E)")
            self._dataset = _UnpairedPatchDataset(
                source_index,
                target_index,
                h5_key=self.hparams.h5_key,
                normalize=self.hparams.normalize,
            )
            total = len(source_index) + len(target_index)
            msg = f"[Dataset] source={len(source_index):,}  target={len(target_index):,}  total={total:,}"
            log.info(msg)
            print(msg, flush=True)

    def _make_loader(self, shuffle: bool) -> DataLoader:
        return DataLoader(
            self._dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=shuffle,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(shuffle=False)
