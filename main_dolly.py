import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from instruct_pipeline import InstructionTextGenerationPipeline

def main():
    base_model = "databricks/dolly-v2-12b"
    load_8bit = False

    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, load_in_8bit=load_8bit, torch_dtype=torch.float16, device_map="auto"
    )

    model.eval()
    if torch.__version__ >= "2":
        model = torch.compile(model)

    print("--PIPELINE INIT--")
    pipe = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    text_prompt = "Write me two lines about Giorgio Armani. In italian."

    print(f"\nPrompt: {text_prompt}\n\n")

    print(pipe(text_prompt))

    text_prompt2 = ""

    
    print(f"\nPrompt: {text_prompt2}\n\n")
    print(pipe(text_prompt2))

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
