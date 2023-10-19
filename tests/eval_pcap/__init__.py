from .evaluate_model import tqdm, eval_model, ImageCaptionMetric

def collate_batch(batch, txt_processor):
    # batch is record
    for item in batch:
        item["comment"] = [txt_processor.pre_caption(cap) for cap in item["comment"]]

    keys = list(batch[0].keys())
    # rec -> key
    # which means from_dl should be FALSE
    batch = {k: [x[k] for x in batch] for k in keys}
    
    return batch

