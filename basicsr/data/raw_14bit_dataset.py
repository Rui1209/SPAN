import numpy as np
from os import path as osp
from torch.utils import data

from basicsr.data.data_util import paths_from_lmdb
from basicsr.utils import FileClient, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Raw14bitDataset(data.Dataset):
    """Read headerless 14-bit RAW dumps (uint16 little-endian) for raw-in raw-out inference."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.lq_folder = opt['dataroot_lq']
        self.raw_size = tuple(opt.get('raw_size', (480, 640)))  # (height, width)
        self.bit_depth = opt.get('bit_depth', 14)
        self.max_val = 2**self.bit_depth - 1
        self.out_channels = opt.get('out_channels', 1)
        if self.out_channels not in (1, 3):
            raise ValueError(f'out_channels must be 1 or 3, got {self.out_channels}')

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder]
            self.io_backend_opt['client_keys'] = ['lq']
            self.paths = paths_from_lmdb(self.lq_folder)
        elif 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.lq_folder, line.rstrip().split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(
                path for path in scandir(self.lq_folder, full_path=True, recursive=True)
                if osp.splitext(path)[1].lower() == '.raw')

        if not self.paths:
            raise FileNotFoundError(f'No RAW files found in {self.lq_folder}')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        lq_path = self.paths[index]
        raw_bytes = self.file_client.get(lq_path, 'lq')
        image = np.frombuffer(raw_bytes, dtype='<u2')
        height, width = self.raw_size
        if image.size != height * width:
            raise ValueError(
                f'Expected {height * width} uint16 words for {width}x{height}, '
                f'got {image.size}: {lq_path}')
        if int(image.max()) > self.max_val:
            raise ValueError(
                f'Pixel value exceeds {self.bit_depth}-bit range ({self.max_val}): {lq_path}')
        image = image.reshape(height, width).astype(np.float32) / self.max_val
        if self.out_channels == 3:
            # repeat mono RAW into 3 identical channels for RGB-pretrained models (e.g. SPAN)
            image = np.repeat(image[..., None], 3, axis=2)
        else:
            image = image[..., None]

        return {
            'lq': img2tensor(np.ascontiguousarray(image), bgr2rgb=False, float32=True),
            'lq_path': lq_path,
        }

    def __len__(self):
        return len(self.paths)
