from collections import OrderedDict
from pathlib import Path
from pytorch_lightning.utilities import rank_zero_only
from torch import nn
import timm

from pointbev.models.img_encoder.backbones.common import Backbone

CKPT_MAP = {"b0": "efficientvit_b0.r224_in1k", "b3": "efficientvit_b3.r224_in1k"}


class EfficientViT(Backbone):
    def __init__(self, checkpoint_path=None, version='b0', downsample=8):
        super().__init__()
        self.version = version

        assert downsample == 8, "EfficientNet only supported for downsample=8"
        self.downsample = downsample
        self._init_efficientvit(checkpoint_path, version)

    def _init_efficientvit(self, weights_path, version):
        if weights_path is not None:
            weights_path = Path(weights_path) / f"{CKPT_MAP[version]}.bin"
            if not weights_path.exists():
                message = f"EfficientViT weights file does not exists at weights_path {weights_path}"
                weights_path = None
            else:
                message = (f"EfficientViT exists and is loaded at weights_path {weights_path}")
                weights_path = str(weights_path)
        else:
            message = "EfficientViT weights file not given, downloading..."

        eff_vit_model = timm.create_model(CKPT_MAP[version], pretrained=True, features_only=True,
            pretrained_cfg_overlay=dict(file=weights_path))
        
        self._stem_in_conv, self._stem_res0, self.stages_0 = (
            eff_vit_model.stem_in_conv, eff_vit_model.stem_res0, eff_vit_model.stages_0
            )
        
        self.stages_1, self.stages_2, self.stages_3 = (
            eff_vit_model.stages_1, eff_vit_model.stages_2, eff_vit_model.stages_3
        )

        del eff_vit_model
        self._print_loaded_file(message)
        
    @rank_zero_only
    def _print_loaded_file(self, message):
        print("# -------- Backbone -------- #")
        print(message, end="\n")

    def forward(self, x, return_all=False):
        x = self.stages_0(self._stem_res0(self._stem_in_conv(x)))
        
        out_1 = self.stages_1(x)
        out_2 = self.stages_2(out_1)
        out_3 = self.stages_3(out_2)

        return OrderedDict({'out_1':out_1, 'out_2':out_2, "out_3":out_3})
    

if __name__ == "__main__":
    import torch
    x = torch.randn((4, 3, 224, 480)).cuda()
    effvit = EfficientViT(checkpoint_path='/mnt/iag/user/daiyiheng/ckpt/backbones/').cuda()
    print(effvit(x))
