import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

def visualize_h5(filepath):
    with h5py.File(filepath, 'r') as f:
        patches = f['patches']
        print(f"Number of patches: {len(patches)}")
        
        # Visualize first 8 patches
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i, ax in enumerate(axes.flatten()):
            # Decode bytes -> image
            patch_bytes = np.array(patches[i]).tobytes()
            patch = Image.open(io.BytesIO(patch_bytes))
            ax.imshow(patch)
            ax.set_title(f"Patch {i}")
            ax.axis('off')
        plt.tight_layout()
        plt.show()

visualize_h5("lung_img_patches/train_B/202509656-7.h5")