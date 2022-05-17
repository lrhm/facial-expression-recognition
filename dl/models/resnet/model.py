from argparse import Namespace
from .modules import ResNetClassifier
from ...base_lightning_modules.base_classification_model import BaseClassificationModel


class Model(BaseClassificationModel):
    def __init__(self, params: Namespace):
        super().__init__(params)

        self.generator = ResNetClassifier(
            num_blocks=[4, 4, 4], c_hidden=[64, 64, 128]
        )
