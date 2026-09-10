import random

import torch
from torchvision.transforms.functional import normalize

from basicsr.data.paired_image_dataset import PairedImageDataset
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedMultiShapeDataset(PairedImageDataset):
    """Paired image dataset for multi-shape training crops.

    Unlike ``PairedImageDataset``, this dataset returns full-size image pairs
    in the train phase. The random crop (one HR shape per batch, randomly
    chosen from ``gt_shapes``) and the flip/rotation augmentation are applied
    in ``paired_multi_shape_collate``, so that all samples in a batch share
    the same shape and can be stacked.

    Args:
        opt (dict): Same as ``PairedImageDataset``, with one extra key:
        gt_shapes (list[list[int]]): Candidate HR crop shapes (h, w). Each
            shape must be divisible by the scale factor.
    """

    def __init__(self, opt):
        super(PairedMultiShapeDataset, self).__init__(opt)
        gt_shapes = opt.get('gt_shapes')
        if not gt_shapes:
            raise ValueError("PairedMultiShapeDataset requires 'gt_shapes': a list of [h, w] pairs.")
        scale = opt['scale']
        for shape in gt_shapes:
            if int(shape[0]) % scale != 0 or int(shape[1]) % scale != 0:
                raise ValueError(f'gt_shape ({shape[0]}, {shape[1]}) is not divisible by scale {scale}.')
        self.gt_shapes = [(int(shape[0]), int(shape[1])) for shape in gt_shapes]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        read_flag = 'grayscale' if self.opt.get('color') == 'gray' else 'color'
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, flag=read_flag, float32=True)
        if img_gt.ndim == 2:
            img_gt = img_gt[..., None]
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, flag=read_flag, float32=True)
        if img_lq.ndim == 2:
            img_lq = img_lq[..., None]

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing
        if self.opt['phase'] != 'train':
            scale = self.opt['scale']
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=read_flag == 'color', float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}


def paired_multi_shape_collate(batch, gt_shapes, scale, use_hflip=True, use_rot=True):
    """Collate paired samples by cropping all of them to one random HR shape.

    A single (h, w) shape is randomly chosen from ``gt_shapes`` per batch and
    clamped to the smallest image in the batch, so every sample has the same
    shape and can be stacked. Each sample is then randomly cropped at aligned
    LQ/GT positions and augmented, mirroring ``paired_random_crop`` and
    ``augment`` in ``basicsr.data.transforms``: hflip and vflip are decided
    per sample (shape-preserving), while the 90-degree rotation is decided
    once per batch so that all samples keep the same shape.

    Args:
        batch (list[dict]): Samples with full-size 'lq'/'gt' tensors (CHW).
        gt_shapes (list[tuple[int]]): Candidate HR crop shapes (h, w).
        scale (int): SR scale factor.
        use_hflip (bool): Use horizontal flips. Default: True.
        use_rot (bool): Use 90-degree rotations. Default: True.

    Returns:
        dict: Stacked 'lq'/'gt' tensors and lists of paths.
    """
    gt_h_min = min(b['gt'].shape[-2] for b in batch)
    gt_w_min = min(b['gt'].shape[-1] for b in batch)
    lq_h_min = min(b['lq'].shape[-2] for b in batch)
    lq_w_min = min(b['lq'].shape[-1] for b in batch)

    gt_h, gt_w = random.choice(gt_shapes)
    gt_h = min(int(gt_h), gt_h_min, lq_h_min * scale)
    gt_w = min(int(gt_w), gt_w_min, lq_w_min * scale)
    gt_h -= gt_h % scale
    gt_w -= gt_w % scale
    lq_h, lq_w = gt_h // scale, gt_w // scale
    if gt_h < scale or gt_w < scale:
        raise ValueError(f'Images in the batch are too small for multi-shape cropping: ({gt_h}, {gt_w}).')

    rot90 = use_rot and random.random() < 0.5
    lq_list, gt_list = [], []
    for sample in batch:
        lq, gt = sample['lq'], sample['gt']
        h_lq, w_lq = lq.shape[-2:]
        h_gt, w_gt = gt.shape[-2:]
        if h_gt < gt_h or w_gt < gt_w or h_lq < lq_h or w_lq < lq_w:
            raise ValueError(f'Image smaller than the crop size ({gt_h}, {gt_w}): {sample["gt_path"]}.')
        top = random.randint(0, h_lq - lq_h)
        left = random.randint(0, w_lq - lq_w)
        lq = lq[..., top:top + lq_h, left:left + lq_w]
        gt = gt[..., top * scale:top * scale + gt_h, left * scale:left * scale + gt_w]

        hflip = use_hflip and random.random() < 0.5
        vflip = use_rot and random.random() < 0.5
        if hflip:
            lq = torch.flip(lq, dims=[-1])
            gt = torch.flip(gt, dims=[-1])
        if vflip:
            lq = torch.flip(lq, dims=[-2])
            gt = torch.flip(gt, dims=[-2])
        if rot90:
            lq = lq.transpose(-2, -1).contiguous()
            gt = gt.transpose(-2, -1).contiguous()
        lq_list.append(lq)
        gt_list.append(gt)

    return {
        'lq': torch.stack(lq_list, dim=0),
        'gt': torch.stack(gt_list, dim=0),
        'lq_path': [b['lq_path'] for b in batch],
        'gt_path': [b['gt_path'] for b in batch]
    }
