import logging
import time
from os import path as osp

import torch

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.test_v1 import get_size_distribution, profile_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    """Run the standard BasicSR test pipeline with the 14-bit extensions."""
    opt, _ = parse_options(root_path, is_train=False)
    torch.backends.cudnn.benchmark = True

    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    model = build_model(opt)
    sizes, mode_size, max_size = get_size_distribution(opt['datasets'])
    if mode_size is not None:
        logger.info('=== LR Size Distribution ===')
        for size, count in sorted(sizes.items(), key=lambda item: -item[1]):
            logger.info(f'  ({size[0]},{size[1]}): {count}')
        logger.info(f'  Mode: ({mode_size[0]},{mode_size[1]}), Max: ({max_size[0]},{max_size[1]})')
        profile_model(model, logger, mode_size, max_size)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.perf_counter() - start
        count = len(test_loader.dataset)
        logger.info(f'Total pipeline time: {total_time:.4f}s ({total_time / count:.4f}s/image)')
        times = getattr(model, '_inference_times', [])
        if times:
            logger.info(f'Average inference time: {sum(times) / len(times):.4f}s/image (pure model)')


if __name__ == '__main__':
    root_path = osp.dirname(osp.abspath(__file__))
    test_pipeline(root_path)
