import os
import time
from os import path as osp
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY


MAX_14BIT = 2**14 - 1
TARGET_SIZE = (1920, 1080)  # (width, height)


@MODEL_REGISTRY.register()
class SRModelRaw(SRModel):
    """SRModel validation with headerless 14-bit RAW output for RAW input."""

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        output_root = osp.join(self.opt['path']['visualization'], dataset_name)
        os.makedirs(output_root, exist_ok=True)
        use_pbar = self.opt['val'].get('pbar', False)
        raw_out = self.opt['val'].get('raw_out', {})
        save_x4 = raw_out.get('save_x4', True)
        save_1080 = raw_out.get('save_1920x1080', True)
        save_preview = raw_out.get('save_preview_png', True)
        inference_times = []
        iterator = tqdm(dataloader, total=len(dataloader), unit='image') if use_pbar else dataloader

        with torch.inference_mode():
            for val_data in iterator:
                self.feed_data(val_data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                self.test()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_times.append(time.perf_counter() - start_time)
                if save_img:
                    source_path = Path(val_data['lq_path'][0]).resolve()
                    input_root = Path(dataloader.dataset.lq_folder).resolve()
                    relative_path = source_path.relative_to(input_root)
                    output_path = Path(output_root) / relative_path
                    self.save_raw(self.output, output_path, save_x4, save_1080, save_preview)
                del self.lq
                del self.output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if use_pbar:
                    iterator.set_description(f'Test {source_path.name}')
        self._inference_times = inference_times
        logger = get_root_logger()
        logger.info(
            f'Validation {dataset_name}: {len(inference_times)} images, '
            f'inference {np.mean(inference_times) * 1000:.1f} ms avg '
            f'({np.min(inference_times) * 1000:.1f}/{np.max(inference_times) * 1000:.1f} ms min/max)')

    @staticmethod
    def save_raw(tensor, output_path, save_x4=True, save_1080=True, save_preview=True):
        """Write model output as headerless 14-bit uint16 little-endian RAW files.

        Output layout (mirrors the input directory structure):
        - ``<stem>.raw``: native x4 output (2560x1920 for 640x480 input)
        - ``<stem>_1920x1080.raw``: 16:9 center crop (2560x1440) followed by an
          isotropic 0.75x downscale to 1920x1080
        - ``<stem>_preview.png``: 8-bit preview (1-99 percentile stretch)
        """
        image = tensor.squeeze(0).detach().float().cpu().clamp_(0, 1)
        image = image.permute(1, 2, 0).numpy()  # HWC
        if image.shape[2] not in (1, 3):
            raise ValueError(
                f'SRModelRaw expects 1- or 3-channel output, got {image.shape[2]} channels')
        # mono output uses the plane directly; 3-channel model output (e.g. SPAN fed
        # with a repeated mono RAW) takes the R channel, matching SRModel14bit.
        plane = image[..., 0]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stem = output_path.stem

        if save_x4:
            np.rint(plane * MAX_14BIT).astype('<u2').tofile(str(output_path))

        if save_1080:
            height, width = plane.shape
            crop_h = TARGET_SIZE[1] * width // TARGET_SIZE[0]
            if crop_h > height:
                raise ValueError(
                    f'Output {width}x{height} too small for a {TARGET_SIZE[0]}x{TARGET_SIZE[1]} crop')
            top = (height - crop_h) // 2
            cropped = plane[top:top + crop_h]
            resized = cv2.resize(cropped, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            np.rint(resized * MAX_14BIT).astype('<u2').tofile(
                str(output_path.with_name(f'{stem}_1920x1080.raw')))
            preview_src = resized
        else:
            preview_src = plane

        if save_preview:
            lo, hi = np.percentile(preview_src, (1, 99))
            preview = np.clip((preview_src - lo) / max(hi - lo, 1e-6), 0, 1)
            preview = np.rint(preview * 255).astype(np.uint8)
            cv2.imwrite(str(output_path.with_name(f'{stem}_preview.png')), preview)
