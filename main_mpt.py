import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from instruct_pipeline import InstructionTextGenerationPipeline

import transformers
def main():
    name = 'mosaicml/mpt-30b-instruct'

    config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
    #config.max_seq_len = 8192
    #config.attn_config['attn_impl'] = 'triton'  # change this to use triton-based FlashAttention
    #config.init_device = 'cuda:0'  # For fast initialization directly on GPU!

    load_8bit = True
    tokenizer = AutoTokenizer.from_pretrained(name)  # , padding_side="left")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        name,
        config=config,
        torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
        trust_remote_code=True,
        load_in_8bit=load_8bit,
        device_map="auto",
    )




    model.eval()
    if torch.__version__ >= "2":
        model = torch.compile(model)

    print("--PIPELINE INIT--")
    pipe = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    text_prompt = "Write me two lines about Giorgio Armani. In italian."

    print(f"\nPrompt: {text_prompt}\n")

    print(pipe(text_prompt))

    text_prompt2 = 'Rimossa MAIL Reclamo'

    torch.cuda.empty_cache()
    
    print(f"\nPrompt: {text_prompt2}\n\n")
    print(pipe(text_prompt2)[0]['generated_text'].split('<|endoftext|>')[0])




if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
