# example call open AI API with GPT-4o on local images
from openai import OpenAI
import base64



import yaml
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to the config YAML file')
parser.add_argument('--image_root',default='mobileworldbench/qa_images', type=str, help='Root directory for images')
args = parser.parse_args()
with open(args.config) as f:
    data = yaml.safe_load(f)
# breakpoint()
client = OpenAI(
    base_url=data['base_url'],
    api_key=data['api_key']  # vLLM doesn’t require authentication
)

model = data['model']
out_name=data['out_name']



# out_name='qwen3_8b_ckpt2000'
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def query(images, prompt):
    """
    images: list of local image paths (e.g., ["cat.jpg", "dog.png"])
    prompt: text instruction for GPT-4o
    """
    image_inputs = [
        # The API expects the `image_url` field to be an object, not a raw string.
        # Wrap the data URL inside an object (key name `url`) so the server sees an object.
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(p)}"}}
        for p in images
    ]

    # Build messages payload so we can inspect it on error
    messages_payload = [
        {"role": "system", "content": "You are a helpful vision assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *image_inputs
            ],
        },
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages_payload,
            temperature=0.0,
            max_tokens=20,
        )
    except Exception as e:
        # On BadRequest from the API we often get helpful clues in the exception message.
        # Print the constructed payload to help debugging (do NOT log API keys).
        print("OpenAI call failed:\n", e)
        try:
            # show the payload structure without large binary content (truncate data URLs)
            safe_messages = []
            for m in messages_payload:
                if isinstance(m, dict):
                    mcopy = m.copy()
                    if "content" in mcopy and isinstance(mcopy["content"], list):
                        content_copy = []
                        for c in mcopy["content"]:
                            if isinstance(c, dict) and c.get("type") == "image_url":
                                # keep only a short prefix of the data URL to avoid huge prints
                                url = c.get("image_url", {}).get("url")
                                if isinstance(url, str):
                                    url = url[:100] + "...[truncated]"
                                content_copy.append({"type": "image_url", "image_url": {"url": url}})
                            else:
                                content_copy.append(c)
                        mcopy["content"] = content_copy
                    safe_messages.append(mcopy)
                else:
                    safe_messages.append(m)
            print("Constructed messages payload (safe preview):\n", safe_messages)
        except Exception:
            pass
        raise

    return response.choices[0].message.content


# Example usage:
# if __name__ == "__main__":
#     result = query(["./example1.jpg", "./example2.png"], "Describe the differences between these two images.")
#     print(result)

PROMPT = "You are an intelligent GUI agent capable of understanding GUIs and actions on mobile devices. Given the current GUI screenshot and input action, answer the following questions based on your predictions of the changes that will occur on the next screen after the action is performed.\n The action is : {action}\n Answer with yes or no. The question is {question}" 

#
def extract_qa_pairs_from_response(response_text):
    retry = 6
    while retry > 0:
        try:
            qa_pairs = []
            lines = response_text.strip().split('\n')
            for line in lines:
                if line.startswith('Q:'):
                    question = line[2:].strip()
                elif line.startswith('A:'):
                    answer = line[2:].strip()
                    qa_pairs.append((question, answer))
            return qa_pairs
        except Exception as e:
            print("Error parsing response, retrying...", e)
            retry -= 1
    return []

def process_row(row):
    row = dict(row)
    img1 = os.path.join(image_root,row['image_start'])
    # img2 =  os.path.join(image_root,row['image_end'])
    # changes = row['changes']
    question = PROMPT.format(action=row['step_instruction'],question=row['question'])
    answer = query([img1], question)
    row['response'] = answer
    return row
    # breakpoint()
import pandas as pd
import os
from tqdm.cli import tqdm
import json
import sys
data = pd.read_csv("benchmark/qa.csv")  
image_root =args.image_root

process_row(data.iloc[0])


all_qa = []
# shuffle rows deterministically
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
correct = 0
total = 1e-6
# parallel processing using concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
with ThreadPoolExecutor(max_workers=128) as executor:
    futures = [executor.submit(process_row, row) for idx, row in tqdm(data.iterrows())]
    pbar =tqdm(as_completed(futures))
    for future in pbar:
        try:
            result = future.result()
            all_qa.append(result)
            acc = result['answer'].lower() in result['response'].lower()
            total += 1
            
            correct += int(acc)
            r = result['answer'],result['response']
            pbar.set_description(f'Avg {correct / total} {r}')
        except Exception as e:
            print(e.__traceback__)
            continue
        
df = pd.DataFrame(all_qa)
df.to_csv(f"outputs/test_result_{out_name}_qa.csv", index=False)
avg_score = (df['response'] == df['answer']).mean()
print(avg_score)
