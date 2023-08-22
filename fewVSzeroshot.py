# import libraries
import dataiku
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

# Load tokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "databricks/dolly-v2-3b",
    padding_side="left")

# Load model

model = AutoModelForCausalLM.from_pretrained(
    "databricks/dolly-v2-3b",
    device_map="auto",
    torch_dtype=torch.bfloat16
).eval()

# Load data

df = dataiku.Dataset("product_reviews").get_dataframe()

# Define target labels & tokens

target_labels = ['positive', 'neutral', 'negative']
target_token_ids = [tokenizer.encode(k)[0] for k in target_labels]

# Define batch size & use GPU if available

BATCH_SIZE = 8
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Zero-shot

# Build the zero-shot prompt
prompt0 = "Decide whether the following product review's sentiment is positive, neutral, or negative.\n\nProduct review:\n{}\nSentiment:"

results0 = None

for i in range(0, len(df), BATCH_SIZE):
    # Instantiate the prompts
    prompts = [prompt0.format(txt) for txt in df["text"][i:i+BATCH_SIZE]]
    
    # Tokenize the prompts and compute the next token probabilities with the model
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids
    with torch.no_grad():
        outputs = model(input_ids.to(DEVICE))
    result = torch.nn.Softmax(dim=-1)(outputs.logits[:, -1, target_token_ids])
    
    if results0 is None:
        results0 = result
    else:
        results0 = torch.cat((results0, result), axis=0)

predicted_token_ids = torch.argmax(results0, axis=1)
predictions0 = [target_labels[i] for i in predicted_token_ids]

scores0_df = pd.DataFrame(
    results0.float().cpu().numpy(),
    columns=[f"proba_{k}" for k in target_labels]
)
df_zeroshot = pd.concat([df, pd.Series(predictions0, name='prediction'), scores0_df], axis=1)

## Few-shot

# Build the prompt with examples
prompt = "Decide whether the following product reviews' sentiment is positive, neutral, or negative."
examples = [
    (
        "I love my new chess board!",
        "positive"
    ),
    (
        "Not what I expected but I guess it'll do",
        "neutral"
    ),
    (
        "I'm so disappointed. The product seemed much better on the website",
        "negative"
    )
]

for example in examples:
    prompt += f"\n\nProduct review:\n{example[0]}\nSentiment:\n{example[1]}"
prompt += "\n\nProduct review:\n{}\nSentiment:\n"

results = None

for i in range(0, len(df), BATCH_SIZE):
    # Instantiate the prompts
    prompts = [prompt.format(txt) for txt in df["text"][i:i+BATCH_SIZE]]

    # Tokenize the prompts and compute the next token probabilities with the model
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids
    with torch.no_grad():
        outputs = model(input_ids.to(DEVICE))
    result = torch.nn.Softmax(dim=-1)(outputs.logits[:, -1, target_token_ids])
    if results is None:
        results = result
    else:
        results = torch.cat((results, result), axis=0)


predicted_token_ids = torch.argmax(results, axis=1)
predictions = [target_labels[i] for i in predicted_token_ids]

scores_df = pd.DataFrame(
    results.float().cpu().numpy(),
    columns=[f"proba_{k}" for k in target_labels]
)

df_fewshot = pd.concat([df, pd.Series(predictions, name='prediction'), scores_df], axis=1)

## Evaluate the results

acc_zeroshot = accuracy_score(df_zeroshot["sentiment"], df_zeroshot["prediction"])
acc_fewshot = accuracy_score(df_fewshot["sentiment"], df_fewshot["prediction"])


f1_zeroshot = f1_score(df_zeroshot["sentiment"], df_zeroshot["prediction"], average="weighted")
f1_fewshot = f1_score(df_fewshot["sentiment"], df_fewshot["prediction"], average="weighted")

print(acc_zeroshot, acc_fewshot, f1_zeroshot, f1_fewshot)





