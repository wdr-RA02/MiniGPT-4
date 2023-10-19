import torch
from typing import List, Optional, Union

from transformers import StoppingCriteriaList
from minigpt4.conversation.conversation import StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

# this supports batch decoding
class MiniGPT4_CLIGenerator(object):
    def __init__(self, model, vis_processor, 
                 instruction:str,  
                 device='cuda:0', 
                 stopping_criteria=None):
        
        self.device = device
        self.model = model
        self.instruction = instruction
        self.vis_processor = vis_processor

        if stopping_criteria is not None:
            self.stopping_criteria = stopping_criteria
        else:
            stop_words_ids = [torch.tensor([2]).to(self.device)]
            self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def encode_img(self, img_list: List[str]):
        '''img_list: list of img paths'''
        batch_size = len(img_list)
        raw_images = [Image.open(image).convert('RGB') for image in img_list]
        image = [self.vis_processor(img) for img in raw_images]
        image = torch.stack(image, dim=0).to(self.device)
        
        image_emb, _ = self.model.encode_img(image)
        # [b, l, d]-> [1, b, l, d]
        if len(image_emb.shape) == 3:
            image_emb.unsqueeze_(0)

        return image_emb, batch_size
    
    def expand_prompt(self, 
                      prompts: List[str]):
        # preprocessing
        for idx in range(len(prompts)):
            if not prompts[idx].startswith("<Img><ImageHere></Img>"):
                prompts[idx] = "<Img><ImageHere></Img> " + prompts[idx].lstrip()
        # insert instruction
        prompt_segs = [self.instruction.format(prompt).split('<ImageHere>')\
                        for prompt in prompts]
        
        return prompt_segs
     
    def get_context_emb(self, prompt_segs: list, image_emb:torch.Tensor):
        '''image_emb : [1, b, p**2+1, d]'''
        bsz_id = 1 if len(image_emb.shape)==4 else 0
        batch_size = image_emb.shape[bsz_id] 
        assert len(prompt_segs) == batch_size, "batchsize mismatch"\
                                               "prompt_list length: {}, " \
                                               "#images: {}".format(len(prompt_segs),image_emb.shape[0])

        seg_tokens = [
            [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", 
                add_special_tokens=i == 0).input_ids.to(self.device)
            # only add bos to the first seg
            for i, seg in enumerate(zip(each_prompt))
            ] for each_prompt in prompt_segs
        ]

        print('debug device: ', self.device)
        print('debug model device: ', self.model.device)

        # print(image_emb[:,0:1].shape)
        seg_embs = [[self.model.embed_tokens(seg_t) for seg_t in seg_token] for seg_token in seg_tokens]
        mixed_embs = [([emb for pair in zip(seg_emb[:-1], image_emb[:,i].unsqueeze(1)) for emb in pair] + [seg_emb[-1]]) 
                      for i, seg_emb in enumerate(seg_embs)]
        # print([a.shape for a in mixed_embs[0]])

        mixed_embs = [torch.cat(mixed_emb, dim=1).to(self.device) for mixed_emb in mixed_embs]

        return mixed_embs

    def post_processing(self, output_texts):
        answers = []
        for txt in output_texts:
            # truncate unclean #'s in output
            txt = txt.split("##")[0]
            # clean errorly generated instr
            txt = txt.split("Assistant:")[-1].strip()
            answers.append(txt)
        
        return answers

    def left_padding(self, input_embeds_list: list):
        '''
        input_embeds_list: a list of input_embeds, length is batch
        '''
        batch = len(input_embeds_list)
        dtype = input_embeds_list[0].dtype
        embed_dim = input_embeds_list[0].shape[-1]
        # try left padding as in commented generate fn
        max_len = max(emb.shape[1] for emb in input_embeds_list)

        embs = torch.zeros([batch, max_len, embed_dim], dtype=dtype, device=self.device)
        attn_mask = torch.zeros([batch, max_len], dtype=torch.int, device=self.device)

        # fill the embedding to the right
        for i, emb in enumerate(input_embeds_list):
            emb_len = emb.shape[1]
            embs[i, -emb_len:] = emb[0]
            attn_mask[i, -emb_len:] = 1
        
        return embs, attn_mask
    
    def generate_response(self,
                         prompts: Union[str, List[str]], 
                         img_list: List[str], 
                         num_beams=1, min_length=1, top_p=0.9, do_sample:bool=False,
                         repetition_penalty=1.05, length_penalty=1, temperature=1.0, max_length=200):
        # prepare images and prompts
        img_embed, bsz = self.encode_img(img_list)
        if isinstance(prompts, str):
            prompts = [prompts] * bsz

        # length equity check
        assert bsz == len(prompts), "List sizes mismatch between img and prompt, "\
                                    "got {} and {} respectively.".format(bsz, len(prompts))
    
        prompts = self.expand_prompt(prompts)

        # prompts= [["<Img>", "</Img> ...."]]
        input_embeds_list = self.get_context_emb(prompts, img_embed)
        # input_emb_list = [e_batch0, ..., e_batchn]
        inputs_embeds, attn_mask = self.left_padding(input_embeds_list)
        print(prompts[0])
        
        # generate kwargs
        generation_kwargs=dict(
            max_length=max_length,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=float(temperature),
        )

        with self.model.maybe_autocast():
            output_token = self.model.llama_model.generate(inputs_embeds=inputs_embeds,
                                                        attention_mask=attn_mask,
                                                        **generation_kwargs)
            output_texts = self.model.llama_tokenizer.batch_decode(output_token, skip_special_tokens=True)

        return output_token, self.post_processing(output_texts)
    
if __name__ == "__main__":
    """
    prompt = "Write a comment of this image."
    instruction = instruction_dict[model_config.model_type]

    generator = BaseCLIGen(model, vis_processor, instruction=instruction, 
                device='cuda:{}'.format(args.gpu_id),
                stopping_criteria=stopping_criteria)
    
    import json
    with open("tests/inference_src.json", "r") as f:
        examples = json.load(f)
    
    # convert record to set
    infer_size = len(examples)
    keys = list(examples[0].keys())
    examples={k: [x[k] for x in examples] for k in keys}
    examples_out=img_hash_to_addr(examples, "dataset/PCap/yfcc_images/","{}.jpg")

    prompts = [prompt] * infer_size
    output, output_text = generator.generate_response(
        prompts, 
        examples_out["images"],
        do_sample=False
    )

    print(output_text)
    """
