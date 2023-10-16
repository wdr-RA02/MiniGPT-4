import os
import logging

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.personality_captions import PCapDataset

@registry.register_builder("personality_captions")
class PCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = PCapDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/p_captions.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'personality_captions/train.json')],
            vis_root=os.path.join(storage_path, 'yfcc_images'),
        )

        return datasets

        