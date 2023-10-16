from datasets import load_dataset
from .utils import img_hash_to_addr, collate_test_set

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class Personality_Captions(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths,
                 max_len:int=30, **kwargs):
        super().__init__()
        # preprocessor: BlipPreprocessor
        self.vis_preprocessor=vis_processor
        self.txt_preprocessor=text_processor

        # dataset: load from datasets.load_dataset
        self.annotation=load_dataset("json", data_files=ann_paths[0], split="train")
        # merge additional column into "comment"
        if "additional_comments" in self.annotation.column_names:
            self.annotation=self.annotation.map(collate_test_set, batch_size=128)
        
        # ann_paths <=> config["img_path"]
        self.img_addr=vis_root
        self.img_name_fmt="{}.jpg"

        # others
        self.max_len=max_len

        # for COCO-like ds compatibility
        # dont know what use for now
        n=0
        self.img_ids={}
        for ann in self.annotation:
            img_id = ann["image_hash"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
    
    # from class BaseDataset
    def collater(self, samples):
        return default_collate(samples)
    
    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __unsqueeze(self, returns:dict, squeeze:bool=False):
        if not squeeze:
            return returns
        
        for k in returns:
            returns[k] = returns[k][0]
        
        return returns

    def __getitem__(self, index):
        squeeze=isinstance(index, int)
        # 对于单int提取的情况则升维
        if squeeze:
            index=[index]

        item=img_hash_to_addr(self.annotation[index], self.img_addr, self.img_name_fmt)
        imgs=[Image.open(img).convert("RGB") for img in item["images"]]

        images=[self.vis_preprocessor(img) for img in imgs]
        captions=[self.txt_preprocessor(text) for text in item["comment"]]

        out = {
            "image": images,
            "answer": captions,
            "personality": item["personality"],
            "image_id": [self.img_ids[hash] for hash in item["image_hash"]],
        }

        return self.__unsqueeze(out, squeeze) 

    def __len__(self):
        return len(self.annotation)


