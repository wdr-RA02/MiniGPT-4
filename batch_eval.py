import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.cli_infer import (MiniGPT4_CLIGenerator as BaseCLIGen, 
                                             StoppingCriteriaSub)
from minigpt4.datasets.datasets.personality_captions.utils import img_hash_to_addr

from tests.eval_pcap import eval_model, collate_batch
from tqdm import tqdm
from functools import partial

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from typing import List, Union


def parse_args():
    parser = argparse.ArgumentParser(description="CLI Evaluator")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

instruction_dict = {
    'pretrain_vicuna0': 
        "Give the following image: <Img>ImageContent</Img>. "
        "You will be able to see the image once I provide it to you. Please answer my questions. "
        "###Human: {} ###Assistant:",
    'pretrain_llama2': 
        "Give the following image: <Img>ImageContent</Img>. "
        "You will be able to see the image once I provide it to you. Please answer my questions. "
        "<s>[INST] {} [/INST] "
}
# add _pcap as suffix
dict_keys = list(instruction_dict.keys())
for key in dict_keys:
    instruction_dict["{}_pcap".format(key)] = instruction_dict[key]

args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

txt_processor_cfg = cfg.datasets_cfg.cc_sbu_align.text_processor.train
txt_processor = registry.get_processor_class(txt_processor_cfg.name).from_config(txt_processor_cfg)

stop_words_ids = [[835], [2277, 29937], [29937]]
stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

print('Initialization Finished')

# this supports batch decoding
class CLIGeneratorForPCap(BaseCLIGen):
    def __init__(self, model, vis_processor, 
                 instruction:str, 
                 debug: bool=False,
                 persona_token:str = "<persona>",
                 device='cuda:0', stopping_criteria=None):
        super().__init__(model, vis_processor, 
                         instruction, device, debug,stopping_criteria)
        self.persona_token = persona_token
    
    def insert_persona(self, 
                       prompts: Union[str, List[str]],
                       personalities: List[str]):
        # preprocessing: insert persona
        bsz = len(personalities)
        if isinstance(prompts, str):
            prompts = [prompts] * bsz

        for idx in range(bsz):
            if not prompts[idx].startswith("<Img><ImageHere></Img>"):
                prompts[idx] = "<Img><ImageHere></Img> " + prompts[idx].lstrip()

        # make sure all prompts include <persona>
        assert all(self.persona_token in prompt for prompt in prompts)

        # replace <persona> with personality
        prompts_persona = [prompt.replace(self.persona_token, persona) \
                           for prompt, persona in zip(prompts, personalities)]
        
        return prompts_persona


if __name__=="__main__":
    '''your code here'''
    instruction = instruction_dict[model_config.model_type]
    # instruction = "###Human: {} ###Assistant:"
    generator = CLIGeneratorForPCap(model, vis_processor, instruction=instruction, 
                             device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
    
    eval_json = "dataset/PCap/personality_captions/test.json"
    image_path = "dataset/PCap/yfcc_images"

    collate_batch = partial(collate_batch, txt_processor=txt_processor)
    result = eval_model(generator, model_config, eval_json, image_path, collate_fn=collate_batch)

    print(result)