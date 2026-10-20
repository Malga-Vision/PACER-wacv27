"""HRNetV2-W18 config for the vendored STLD backbone.

Values are IDENTICAL to the original github.com/jhb86253817/STLD
lib/hrnetv2_config.py (itself from Microsoft HRNet). The only change is the
container: the upstream uses `yacs.config.CfgNode`, which is not installed here,
so we provide a minimal attribute+dict-access shim with the same field values.
`hrnetv2.py` (the backbone) is vendored byte-exact and reads these fields both as
attributes (cfg.MODEL.EXTRA.STAGE2) and as dicts (layer_config['NUM_CHANNELS']).
"""
from __future__ import absolute_import, division, print_function


class CN(dict):
    """Minimal CfgNode shim: dict with attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


_C = CN()

_C.OUTPUT_DIR = 'output'
_C.LOG_DIR = 'log'
_C.GPUS = (0,)
_C.WORKERS = 16
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True

_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

_C.MODEL = CN()
_C.MODEL.NAME = 'hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 68
_C.MODEL.TARGET_TYPE = 'Gaussian'
_C.MODEL.IMAGE_SIZE = [256, 256]
_C.MODEL.HEATMAP_SIZE = [64, 64]
_C.MODEL.SIGMA = 1.5
_C.MODEL.EXTRA = CN()

_C.MODEL.EXTRA.PRETRAINED_LAYERS = ['*']
_C.MODEL.EXTRA.STEM_INPLANES = 64
_C.MODEL.EXTRA.FINAL_CONV_KERNEL = 1
_C.MODEL.EXTRA.WITH_HEAD = True

_C.MODEL.EXTRA.STAGE2 = CN()
_C.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
_C.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
_C.MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
_C.MODEL.EXTRA.STAGE2.NUM_CHANNELS = [18, 36]
_C.MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

_C.MODEL.EXTRA.STAGE3 = CN()
_C.MODEL.EXTRA.STAGE3.NUM_MODULES = 4
_C.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
_C.MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
_C.MODEL.EXTRA.STAGE3.NUM_CHANNELS = [18, 36, 72]
_C.MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

_C.MODEL.EXTRA.STAGE4 = CN()
_C.MODEL.EXTRA.STAGE4.NUM_MODULES = 3
_C.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
_C.MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
_C.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [18, 36, 72, 144]
_C.MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'

_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'AFLW'
_C.DATASET.TRAINSET = ''
_C.DATASET.TESTSET = ''
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30

_C.TRAIN = CN()
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [30, 50]
_C.TRAIN.LR = 0.0001
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.0
_C.TRAIN.WD = 0.0
_C.TRAIN.NESTEROV = False
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 60
_C.TRAIN.RESUME = True
_C.TRAIN.CHECKPOINT = ''
_C.TRAIN.BATCH_SIZE_PER_GPU = 16
_C.TRAIN.SHUFFLE = True

_C.TEST = CN()
_C.TEST.BATCH_SIZE_PER_GPU = 32


def update_config(cfg, args_cfg=None):
    """No-op shim (upstream merged a YAML; we set NUM_JOINTS directly in code)."""
    return cfg
