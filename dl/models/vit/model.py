from argparse import Namespace
from torch.functional import F
from ...base_lightning_modules.base_classification_model import BaseClassificationModel
from vit_pytorch import ViT

class Model(BaseClassificationModel):
    def __init__(self, params: Namespace):
        super().__init__(params)

        self.generator = ViT(
            image_size=48,
            patch_size=4,
            num_classes=7,
            dim=32,
            depth=6,
            heads=8,
            mlp_dim=256,
            channels=1,
        )

    def forward(self, x):
        x = self.generator(x)
        return F.softmax(x, dim=1)

    