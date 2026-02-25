import math
import random
import torch
from PIL import Image
import blobfile as bf

import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=False,
    use_heatmap=False,
    use_Norm=False,
    use_tile=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    csv_path=os.path.join(data_dir,'app_traffic_orig_max_values.csv')
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=0,
        num_shards=1,
        random_crop=random_crop,
        random_flip=random_flip,
        csv_path=csv_path,
        use_heatmap=use_heatmap,
        use_Norm=use_Norm,
        use_tile=use_tile,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        if bf.isdir(full_path) and entry.isdigit():  # 检查是否是名为数字的文件夹
            folder_index = entry
            gasf_name = f"cross_gasf_{folder_index}.tiff"
            mask_name = f"mask_{folder_index}.png"

            gasf_path = bf.join(full_path, gasf_name)
            mask_path = bf.join(full_path, mask_name)

            # 确保两个文件都存在
            if bf.exists(gasf_path) and bf.exists(mask_path):
                results.append((gasf_path, mask_path))
            else:
                print(f"Warning: Missing file in {full_path} — gasf: {bf.exists(gasf_path)}, mask: {bf.exists(mask_path)}")
        elif bf.isdir(full_path):
            # 递归向下查找
            results.extend(_list_image_files_recursively(full_path))
    return results
"""def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results"""

"""
class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict"""
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import os
import tifffile

class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,  # [(gasf_path, mask_path), ...]
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=False,
        csv_path=None,
        use_heatmap=False,
        use_Norm=False,
        use_tile=False
    ):
        super().__init__()
        # 保证按 shard 分配后只保留 (gasf, mask) 对
        self.resolution = resolution
        self.local_image_pairs = image_paths[shard:][::num_shards]
        self.local_classes = None 
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.csv_path=csv_path
        if csv_path is not None:
           
            self.orig_app_max_list = None
        self.heatmap = np.load( "/home/yilai/projects/poster/NetDiffus/heatmap.npy")
        self.use_heatmap =use_heatmap
        self.use_Norm=use_Norm
        self.use_tile=use_tile
        if self.use_tile:
            for gasf_path, _ in self.local_image_pairs:
                gasf = tifffile.imread(gasf_path).astype(np.float32)     # [H, W]
                gasf = np.expand_dims(gasf, axis=-1)                     # [H, W, 1]
                gasf = np.transpose(gasf, [2, 0, 1])                     # [C, H, W]
                base_mask = gasf > 0                                     # bool, [C, H, W]

                # （可选）如果需要“只保留最左侧 1/20 活跃列”，在这里做一次即可：
                col_active = base_mask.any(axis=(0, 1))                  # [W]
                active_idx = np.where(col_active)[0]
                self.tile_mask = None
                if active_idx.size > 0:
                    keep_cols = max(1, int(np.ceil(active_idx.size / 20.0)))
                    leftmost_cols = active_idx[:keep_cols]
                    keep_mask = np.zeros((1, 1, base_mask.shape[2]), dtype=bool)
                    keep_mask[..., leftmost_cols] = True
                    base_mask = base_mask & keep_mask
                self.tile_mask = base_mask.astype(np.float32)  # float32, [C, H, W]
                break
    def __len__(self):
        return len(self.local_image_pairs)

    def __getitem__(self, idx):
        gasf_path, mask_path = self.local_image_pairs[idx]

        gasf_image = tifffile.imread(gasf_path)
   
        with bf.BlobFile(mask_path, "rb") as f:
            mask_image = Image.open(f)
            mask_image.load()
        # 保证 mask 是单通道
        mask_image = mask_image.convert("L")
        if self.random_crop:
            i, j, h, w = self._get_random_crop_coords(gasf_image)
            gasf_image = gasf_image.crop((j, i, j + w, i + h))
            mask_image = mask_image.crop((j, i, j + w, i + h))
        #else:
            #gasf_image = self._center_crop(gasf_image)
            #mask_image = self._center_crop(mask_image)

        if self.random_flip and random.random() < 0.5:
            gasf_image = gasf_image.transpose(Image.FLIP_LEFT_RIGHT)
            mask_image = mask_image.transpose(Image.FLIP_LEFT_RIGHT)
        ##### it is better noticed that the viridis has color range from 1 to 253 instead of 0 to 255
        # Convert to float32 np arrays
        
        gasf_arr = (np.array(gasf_image).astype(np.float32))
        mask_arr = np.array(mask_image).astype(np.float32)/255   # normalize to [0, 1]
        gasf_arr=np.expand_dims(gasf_arr,axis=-1)
        # Transpose gasf to (C, H, W); mask stays (H, W)
        # 计算每行的最大值（在转置前计算，此时形状是 [H, W, 1]）
        row_max_values = np.max(gasf_arr, axis=1)  # 对宽度维度取最大值
        row_max_values = row_max_values.squeeze()  # 移除最后的通道维度，变成 [H]
        
        # 转置到标准格式
        gasf_arr = np.transpose(gasf_arr, [2, 0, 1])
        
        Dict = {}

        if self.use_heatmap:

            shape = gasf_arr.shape
            # 确保heatmap是3维的 (C,H,W)，如果不是则扩展维度
            if len(self.heatmap.shape) == 2:
                self.heatmap = np.expand_dims(self.heatmap, axis=0)
            
            # 执行maxnorm归一化
            max_val = np.max(self.heatmap)
            if max_val > 0:  # 避免除以0
                self.heatmap = self.heatmap / max_val
            
            # 计算需要padding的尺寸
            pad_height = max(0, shape[1] - self.heatmap.shape[1])
            pad_width = max(0, shape[2] - self.heatmap.shape[2])
            
            # 执行padding
            heatmap_resized = np.pad(
                self.heatmap,
                ((0,0), (0,pad_height), (0,pad_width)),
                'constant',
                constant_values=0
            )
            Dict['heatmap'] = torch.from_numpy(heatmap_resized)
        if self.use_Norm:
            norm_img = tifffile.imread("/home/yilai/projects/poster/NetDiffus/tiff_log/_norm/norm_img.tiff")
            #norm_img = (norm_img.astype(np.float32))*2 - 1
            norm_img = np.expand_dims(norm_img, axis=-1)
            norm_img = np.transpose(norm_img, [2, 0, 1])
            Dict['norm_img'] = torch.from_numpy(norm_img)
        # 添加行最大值到字典中
        if self.use_tile:
    
            Dict['tile_mask'] = torch.from_numpy(self.tile_mask)
        Dict['traffic'] = torch.from_numpy(row_max_values)
        return torch.from_numpy(gasf_arr), torch.from_numpy(mask_arr), torch.tensor(gasf_arr.max()), Dict

    def _get_random_crop_coords(self, image):
        width, height = image.size
        th, tw = self.resolution, self.resolution
        if width == tw and height == th:
            return 0, 0, th, tw

        i = random.randint(0, height - th)
        j = random.randint(0, width - tw)
        return i, j, th, tw

    def _center_crop(self, image):
        width, height = image.size
        new_width = new_height = self.resolution
        left = int((width - new_width) / 2)
        top = int((height - new_height) / 2)
        right = left + new_width
        bottom = top + new_height
        return image.crop((left, top, right, bottom))

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
