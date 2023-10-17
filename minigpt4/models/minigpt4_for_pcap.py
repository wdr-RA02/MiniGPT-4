import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.base_model import disabled_train
from minigpt4.models.minigpt_base import MiniGPTBase
from minigpt4.models.Qformer import BertConfig, BertLMHeadModel
from minigpt4.models.minigpt4 import MiniGPT4

BLIP_PT_PATH = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"


@registry.register_model("minigpt4_for_pcap")
class MiniGPT4ForPCap(MiniGPT4):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna0": "configs/models/minigpt4_vicuna0.yaml",
        "pretrain_llama2": "configs/models/minigpt4_llama2.yaml",
    }

    '''
    what to rewrite:
    1. change the <persona> item in the prompts to the specific persona in the prompts
    we adopt pre-injection in ``preparing_embedding`` fn
    2. (maybe) add random shuffle and prompt the model to redraft (as in PoliteFlamingo)

    '''
    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model=BLIP_PT_PATH,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        has_qformer=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        persona_token:str="<persona>",
        **kwargs
    ):
        super.__init__(vit_model=vit_model,
        q_former_model=q_former_model,
        img_size=img_size,
        drop_path_rate=drop_path_rate,
        use_grad_checkpoint=use_grad_checkpoint,
        vit_precision=vit_precision,
        freeze_vit=freeze_vit,
        has_qformer=has_qformer,
        freeze_qformer=freeze_qformer,
        num_query_token=num_query_token,
        llama_model=llama_model,
        prompt_path=prompt_path,
        prompt_template=prompt_template,
        max_txt_len=max_txt_len,
        end_sym=end_sym,
        low_resource=low_resource,  # use 8 bit and put vit in cpu
        device_8bit=device_8bit)

        self.persona_token = kwargs.get("persona_token", "<persona>")

    def preparing_embedding(self, samples):
        ### prepare input tokens
        if 'image' in samples:
            img_embeds, img_atts = self.encode_img(samples["image"])
        else:
            img_embeds = img_atts = None
        
        if self.prompt_list:
            instruction = random.choice(self.prompt_list)
        else:
            instruction = None
        
        # in theory a prompt list must exist here
        assert instruction is not None

        # expand instruction to personas bundled with each img
        inst_persona = [instruction] * len(img_embeds)
        for idx, persona in enumerate(samples["personality"]):
            inst_persona[idx] = inst_persona[idx].replace(self.persona_token, persona)
        
        cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, inst_persona)

        return cond_embeds, cond_atts

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        has_qformer = cfg.get("has_qformer", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        # add
        persona_token = cfg.get("persona_token", "<persona>")

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            has_qformer=has_qformer,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            # add
            persona_token=persona_token,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load MiniGPT-4 Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model