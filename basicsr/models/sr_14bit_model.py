import os
from os import path as osp
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY


MAX_14BIT = 2**14 - 1


@MODEL_REGISTRY.register()
class SRModel14bit(SRModel):
    """SRModel validation with uint16 TIFF output for 14-bit input."""

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        output_root = osp.join(self.opt['path']['visualization'], dataset_name)
        os.makedirs(output_root, exist_ok=True)
        use_pbar = self.opt['val'].get('pbar', False)
        iterator = tqdm(dataloader, total=len(dataloader), unit='image') if use_pbar else dataloader

        with torch.inference_mode():
            for val_data in iterator:
                self.feed_data(val_data)
                self.test()
                if save_img:
                    source_path = Path(val_data['lq_path'][0]).resolve()
                    input_root = Path(dataloader.dataset.lq_folder).resolve()
                    relative_path = source_path.relative_to(input_root)
                    output_path = Path(output_root) / relative_path
                    save_tiff(self.output, output_path, bool(val_data['is_grayscale'][0]))
                del self.lq
                del self.output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if use_pbar:
                    iterator.set_description(f'Test {source_path.name}')


def save_tiff(tensor, output_path, is_grayscale):
    image = tensor.squeeze(0).detach().float().cpu().clamp_(0, 1)
    image = image.permute(1, 2, 0).numpy()
    image = np.rint(image * MAX_14BIT).astype(np.uint16)
    if is_grayscale:
        image = image[..., 0]
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise IOError(f'Unable to write TIFF image: {output_path}')
