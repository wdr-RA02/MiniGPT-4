import torch

from tqdm import tqdm

from datasets import load_dataset
from torch.utils.data import DataLoader

from .eval_metric import ImageCaptionMetric
from minigpt4.datasets.datasets.personality_captions.utils import img_hash_to_addr


def collate_test_set(src):
    '''
    merges additional_comments into comment
    '''

    assert "additional_comments" in src.keys()
    other_keys=[k for k in src.keys() if k not in ["comment","additional_comments"]]
    single_instance = isinstance(src["comment"],str)
    # other elements
    tgt={k:src[k] for k in other_keys}
    if single_instance:
        tgt["comment"]=[src["comment"], *src["additional_comments"]]
    else:
        tgt["comment"]=[items for items in zip(src["comment"], *src["additional_comments"])]

    return tgt


@torch.no_grad()
def eval_model(generator,
               model_cfg, eval_ds_path: str,
               img_addr: str, img_fmt:str = "{}.jpg",
               collate_fn=None):
    
    # load dataset
    test_ds = load_dataset("json", data_files=eval_ds_path, split="train")
    if "additional_comments" in test_ds.column_names:
        print("Collating test dataset...")
        test_ds=test_ds.map(collate_test_set, batch_size=128).remove_columns("additional_comments")

    print("\n------Enter evaluation------")

    # all in minigpt4_metric_pcap.yaml
    prompt=model_cfg.prompt
    batch=model_cfg.eval_batch_size
    num_workers=model_cfg.eval_num_workers

    # from examination, all generated tokens are clean
    # so no token is required
    eos_token=""
    pad_token=""

    evaluator=ImageCaptionMetric(mul_100=True, eos_token=eos_token, pad_token=pad_token)
    test_ds=DataLoader(test_ds, batch_size=batch, 
                       shuffle=False, pin_memory=True,
                       num_workers=num_workers, 
                       collate_fn=collate_fn)
    
    steps = len(test_ds)
    for i, test_item in tqdm(enumerate(test_ds), \
                             total=steps, desc="Processing test set"):
        # generate
        out_item = img_hash_to_addr(test_item, img_addr, img_fmt)
        # prompt insert persona
        prompts = generator.insert_persona(prompt, out_item["personality"])
        _, output_text = generator.generate_response(prompts, 
                                                     out_item["images"],
                                                     do_sample=False,
                                                     max_length=model_cfg.max_txt_len)
        
        evaluator.add(output_text, out_item, batch_id="b%d" % i, from_dl=False)

    result=evaluator.evaluate()
    print("\n------Finished evaluation------")
    # round the results
    result={k:round(v, 3) for k,v in result.items()}

    return result
