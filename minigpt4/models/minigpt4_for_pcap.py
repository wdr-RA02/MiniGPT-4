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
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

BLIP_PT_PATH = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"


@registry.register_model("minigpt4_for_pcap")
class MiniGPT4ForPCap(MiniGPT4):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna0": "configs/models/minigpt4_vicuna0.yaml",
        "pretrain_llama2": "configs/models/minigpt4_llama2.yaml",
        "pretrain_vicuna0_pcap": "configs/models/minigpt4_vicuna0_pcap.yaml",
        "pretrain_llama2_pcap": "configs/models/minigpt4_llama2_pcap.yaml",
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
        # add lora
        lora_r=0,
        lora_alpha=16,
        **kwargs
    ):
        super().__init__(vit_model=vit_model,
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

        self.persona_token = persona_token
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        # load lora
        self.init_lora()

    # copied from base_model.py
    def init_lora(self, lora_target_modules=["q_proj","v_proj"], **lora_kargs):
        if self.lora_r > 0:
            logging.info("Init Lora, rank={}, alpha={}, modules={}".format(self.lora_r,
                                                                           self.lora_alpha,
                                                                           lora_target_modules))
            self.llama_model = prepare_model_for_int8_training(self.llama_model)
            loraconfig = LoraConfig(
                r=self.lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_target_modules,
                lora_dropout=0.05,
                **lora_kargs
            )
            self.llama_model = get_peft_model(self.llama_model, loraconfig)

            self.llama_model.print_trainable_parameters()
            logging.info('Init Lora Done')
        else:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
    

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
        ### prepare target tokens
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["answer"]]

        regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(self.device)

        regress_token_ids = regress_tokens.input_ids
        regress_atts = regress_tokens.attention_mask
        part_targets = regress_token_ids.masked_fill(
            regress_token_ids == self.llama_tokenizer.pad_token_id, -100
        )

        regress_embeds = self.embed_tokens(regress_token_ids)

        return cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets

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
        lora_r = cfg.get("lora_r", 0)
        lora_alpha = cfg.get("lora_alpha", 16)

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
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load MiniGPT-4 Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model