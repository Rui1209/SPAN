import glob
import logging
import time
from os import path as osp

import torch
from PIL import Image
from torch.nn.parallel import DataParallel, DistributedDataParallel

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


IMAGE_EXTENSIONS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')


def get_bare_model(net):
    if isinstance(net, (DataParallel, DistributedDataParallel)):
        return net.module
    return net


def get_size_distribution(dataset_options):
    sizes = {}
    for dataset_opt in dataset_options.values():
        root = dataset_opt.get('dataroot_lq')
        if not root:
            continue
        files = []
        for extension in IMAGE_EXTENSIONS:
            files.extend(glob.glob(osp.join(root, '**', extension), recursive=True))
            files.extend(glob.glob(osp.join(root, '**', extension.upper()), recursive=True))
        for filename in files:
            with Image.open(filename) as image:
                width, height = image.size
            sizes[(height, width)] = sizes.get((height, width), 0) + 1
    if not sizes:
        return {}, None, None
    mode_size = max(sizes, key=sizes.get)
    max_size = max(sizes, key=lambda size: size[0] * size[1])
    return sizes, mode_size, max_size


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def profile_model(model, logger, mode_size, max_size):
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    net = get_bare_model(model.net_g)
    net.eval()
    if hasattr(net, 'deploy'):
        net.deploy()

    params = sum(parameter.numel() for parameter in net.parameters())
    trainable = sum(parameter.numel() for parameter in net.parameters() if parameter.requires_grad)
    weights_bytes = sum(parameter.numel() * parameter.element_size() for parameter in net.parameters())
    logger.info('=== Model Efficiency ===')
    logger.info(f'Total parameters: {params:,d} ({params / 1e6:.4f} M)')
    logger.info(f'Trainable parameters: {trainable:,d}')
    logger.info(f'Model weights memory: {weights_bytes / 1024 ** 2:.2f} MB')

    device = next(net.parameters()).device
    in_chans = model.opt['network_g'].get('num_in_ch', 3)
    dummy = torch.rand(1, in_chans, mode_size[0], mode_size[1], device=device)
    with torch.no_grad():
        analysis = FlopCountAnalysis(net, dummy)
    logger.info(f'FLOPs (mode {mode_size[1]}x{mode_size[0]}): {analysis.total() / 1e9:.4f} GFLOPs')
    logger.info(flop_count_table(analysis))

    if max_size != mode_size:
        max_dummy = torch.rand(1, in_chans, max_size[0], max_size[1], device=device)
        with torch.no_grad():
            max_analysis = FlopCountAnalysis(net, max_dummy)
        logger.info(f'FLOPs (max {max_size[1]}x{max_size[0]}): {max_analysis.total() / 1e9:.4f} GFLOPs')

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)
    with torch.no_grad():
        net(dummy)
    peak_memory = torch.cuda.max_memory_allocated(device=device) / 1024 ** 2 if torch.cuda.is_available() else 0.0
    logger.info(f'Peak inference memory: {peak_memory:.2f} MB ({mode_size[1]}x{mode_size[0]}, batch=1)')

    warmup, runs = 10, 100
    with torch.no_grad():
        for _ in range(warmup):
            net(dummy)
        sync_cuda()
        start = time.perf_counter()
        for _ in range(runs):
            net(dummy)
        sync_cuda()
    logger.info(f'FPS: {runs / (time.perf_counter() - start):.4f} (input {mode_size[1]}x{mode_size[0]}, batch=1)')


def test_pipeline(root_path):
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
        test_loaders.append(build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None,
            seed=opt['manual_seed']))
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")

    sizes, mode_size, max_size = get_size_distribution(opt['datasets'])
    if mode_size is not None:
        logger.info('=== LR Size Distribution ===')
        for size, count in sorted(sizes.items(), key=lambda item: -item[1]):
            logger.info(f'  ({size[0]},{size[1]}): {count}')
        logger.info(f'  Mode: ({mode_size[0]},{mode_size[1]}), Max: ({max_size[0]},{max_size[1]})')

    model = build_model(opt)
    if mode_size is not None:
        profile_model(model, logger, mode_size, max_size)

    for test_loader in test_loaders:
        dataset_name = test_loader.dataset.opt['name']
        logger.info(f'=== Test: {dataset_name} ===')
        sync_cuda()
        start = time.perf_counter()
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])
        sync_cuda()
        total_time = time.perf_counter() - start
        count = len(test_loader.dataset)
        logger.info(f'Total pipeline time: {total_time:.4f}s ({total_time / count:.4f}s/image)')
        times = getattr(model, '_inference_times', [])
        if times:
            logger.info(f'Average inference time: {sum(times) / len(times):.4f}s/image (pure model)')


if __name__ == '__main__':
    test_pipeline(osp.abspath(osp.join(__file__, osp.pardir, osp.pardir)))
