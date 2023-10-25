from batch_eval import *

def test_gen(src_filename = "tests/inference_src.json"):
    prompt = "Please briefly comment this photo as a person with the personality of: <persona>."

    instruction = instruction_dict[model_config.model_type]
    # instruction = "###Human: {} ###Assistant:"
    generator = CLIGeneratorForPCap(model, vis_processor, instruction=instruction, 
                             device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
    import json
    with open(src_filename, "r") as f:
        examples = json.load(f)
    
    # convert record to set
    num_items=len(examples)
    keys = examples[0].keys()
    examples_out = {k: [x[k] for x in examples] for k in keys}
    examples_out = img_hash_to_addr(examples_out, "dataset/PCap/yfcc_images/","{}.jpg")

    prompts = generator.insert_persona(prompt, examples_out["personality"])
    output, output_text = generator.generate_response(prompts, 
                                                      examples_out["images"],
                                                      do_sample=False,
                                                      max_length=50)
    
    # return to record
    record=[{k: examples_out[k][i] for k in keys} for i in range(num_items)]
    for i, item in enumerate(record):

        item.update({"output": output_text[i]})

    print(*record)
    from datetime import datetime as dt
    now_str=dt.strftime(dt.now(), "%Y%m%d-%H%M%S")
    output_filename="minigpt4/output/output_{}.json".format(now_str)
    
    with open(output_filename, "w") as f:
        json.dump(record, f, indent=4)

    print("Saved to {}".format(output_filename))

if __name__=="__main__":
    test_gen()