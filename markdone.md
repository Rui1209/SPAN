# SPANV2 grayscale integration

## Scope

- Import the official SPANV2 architecture into the existing BasicSR project.
- Train a single-channel, 8-bit grayscale x4 model from random initialization.
- Preserve the existing SPAN architecture and existing user changes.

## Implementation checklist

- [x] Add `basicsr/archs/spanv2_arch.py` with `Conv3XC`, `SPABV2`, and `SPANV2_ESR`.
- [x] Register `SPANV2_ESR` through BasicSR architecture auto-discovery.
- [x] Support `num_in_ch=1` and `num_out_ch=1`.
- [x] Use PyTorch attention fallback by default (`use_span_attn: false`).
- [x] Keep `img_range` and `rgb_mean` unused, matching the supplied official SPANV2 forward path.
- [x] Add grayscale loading to `PairedImageDataset`.
- [x] Match paired files by stem while allowing JPEG/PNG extension differences.
- [x] Add `options/train/SPANV2/train_SPANV2_x4_gray.yml` for from-scratch training.
- [x] Add `options/test/SPANV2/test_SPANV2_x4_gray.yml` for paired grayscale testing.
- [x] Run CPU model forward/backward smoke test.
- [x] Run Python syntax compilation for modified modules.
- [x] Run grayscale paired dataset smoke test with JPEG LR and PNG HR.
- [x] Resume training with `--auto_resume` confirmed working on GPU (RTX 3060, PyTorch 2.10.0+cu126).
- [x] Add `basicsr/archs/spanv2_residual_arch.py` with `SPABV2_Residual` and `SPANV2_ESR_Residual`.
- [x] Register `SPANV2_ESR_Residual` through BasicSR architecture auto-discovery.
- [x] Add `options/train/SPANV2/train_SPANV2_x4_gray_residual.yml` for from-scratch training.
- [x] Run CPU forward/backward smoke test for `SPANV2_ESR_Residual`.
- [x] Verify both `SPANV2_ESR` and `SPANV2_ESR_Residual` coexist in registry.
- [ ] Run GPU smoke test after CUDA PyTorch is installed.

## Data contract

- Input formats: JPEG and PNG.
- Input type: 8-bit grayscale.
- Pairing: matching file stems in LR and HR folders.
- Scale: HR dimensions are 4x LR dimensions.
- Model tensors: one channel, float32, `[0, 1]`.

## Initial training command

```bash
python basicsr/train.py -opt options/train/SPANV2/train_SPANV2_x4_gray.yml
```

Replace the four `YourDataset` paths in the YAML before training.

## Resume training command

```bash
python basicsr/train.py -opt options/train/SPANV2/train_SPANV2_x4_gray.yml --auto_resume
```

Keep the same `name` and do not set `pretrain_network_g` so BasicSR restores the
latest `.state` under `experiments/<name>/training_states/`.

## Testing command

```bash
python basicsr/test.py -opt options/test/SPANV2/test_SPANV2_x4_gray.yml
```

Fill in `dataroot_gt` and `dataroot_lq` in the YAML first. The test pipeline uses
`PairedImageDataset` with `color: gray`, loads `params_ema` from
`net_g_300000.pth`, computes PSNR/SSIM with `crop_border: 4`, and saves SR images
under `experiments/SPANV2_x4_gray_test/visualization/CustomTest/`.

## SPANV2_ESR_Residual variant

Attention moved from after the 3rd 3x3 conv to after the 1st 3x3 conv.
Uses `(x + f1) * sign(m) * |m|^0.8` as the guidance mechanism.
block_2~5 output `f3` without ReLU; block_1 outputs `act(f3)`.

### Architecture file

`basicsr/archs/spanv2_residual_arch.py` (auto-discovered by BasicSR).

### Training command

```bash
python basicsr/train.py -opt options/train/SPANV2/train_SPANV2_x4_gray_residual.yml
```

### Resume command

```bash
python basicsr/train.py -opt options/train/SPANV2/train_SPANV2_x4_gray_residual.yml --auto_resume
```

### Smoke test results

- `py_compile` passed.
- CPU forward: `[1, 1, 8, 10] -> [1, 1, 32, 40]`.
- CPU backward: gradients OK.
- Parameters: 134,880 (same as `SPANV2_ESR`).
- Both architectures registered and coexist.

## Validation performed

- `py_compile` passed for the modified Python modules.
- CPU model test passed: `[1, 1, 8, 10] -> [1, 1, 32, 40]`.
- CPU backward test passed with non-null input gradients.
- GPU test is pending because the current environment reports `torch 2.1.1+cpu` and no CUDA device.
