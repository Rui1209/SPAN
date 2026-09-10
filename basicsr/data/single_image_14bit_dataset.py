import cv2
import numpy as np
from os import path as osp
from torch.utils import data

from basicsr.data.data_util import paths_from_lmdb
from basicsr.utils import FileClient, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY


MAX_14BIT = 2**14 - 1


@DATASET_REGISTRY.register()
class SingleImage14bitDataset(data.Dataset):
    """Read 14-bit TIFF images for inference without changing 8-bit datasets."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.lq_folder = opt['dataroot_lq']
        self.out_channels = opt.get('out_channels', 3)
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
                if osp.splitext(path)[1].lower() in ('.tif', '.tiff'))

        if not self.paths:
            raise FileNotFoundError(f'No TIFF files found in {self.lq_folder}')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        lq_path = self.paths[index]
        image_bytes = self.file_client.get(lq_path, 'lq')
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f'Unable to read TIFF image: {lq_path}')
        if image.dtype != np.uint16:
            raise ValueError(f'Expected uint16 14-bit TIFF, got {image.dtype}: {lq_path}')
        if np.max(image) > MAX_14BIT:
            raise ValueError(f'Pixel value exceeds 14-bit range ({MAX_14BIT}): {lq_path}')

        is_grayscale = image.ndim == 2 or image.shape[-1] == 1
        if image.ndim == 2:
            image = image[..., None]
        if image.ndim != 3 or image.shape[2] not in (1, 3):
            raise ValueError(f'Expected grayscale or 3-channel TIFF, got {image.shape}: {lq_path}')

        image = image.astype(np.float32) / MAX_14BIT
        if image.shape[2] == 1:
            if self.out_channels == 3:
                image = np.repeat(image, 3, axis=2)
        else:
            if self.out_channels == 1:
                raise ValueError(f'out_channels=1 requires grayscale TIFF, got {image.shape}: {lq_path}')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return {
            'lq': img2tensor(np.ascontiguousarray(image), bgr2rgb=False, float32=True),
            'lq_path': lq_path,
            'is_grayscale': is_grayscale,
        }

    def __len__(self):
        return len(self.paths)
