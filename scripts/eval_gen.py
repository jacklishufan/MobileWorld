# example call open AI API with GPT-4o on local images
from openai import OpenAI
import base64


import yaml
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to the config YAML file')
parser.add_argument('--image_root',default='mobileworldbench/gen_images', type=str, help='Root directory for images')
args = parser.parse_args()
with open(args.config) as f:
    data = yaml.safe_load(f)

client = OpenAI(
    base_url=data['base_url'],
    api_key=data['api_key']  # vLLM doesn’t require authentication
)

model = data['model']
out_name=data['out_name']
from PIL import Image
import io
def encode_image(image_path):
    
    # Open and convert image to RGB
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        bytes_data = buffer.getvalue()
        
    return base64.b64encode(bytes_data).decode('utf-8')

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
        # {"role": "system", "content": "You are a helpful vision assistant."},
        {
            "role": "user",
            "content": [
                *image_inputs,
                {"type": "text", "text": prompt},

            ],
        },
    ]

    try:
        response = client.chat.completions.create(
            model=model,  # or "gpt-4o" for full version
            messages=messages_payload,
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



PROMPT =  "You are an intelligent GUI agent capable of understanding GUIs and actions on mobile devices. Given the current GUI screenshot and input action, describe the changes that will occur on the next screen after the action is performed.\n The action is {action}." 

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
    img1 = os.path.join(args.image_root,row['input_image'])
    question = PROMPT.format(action=row['action'])
    # gt df['change_desc']
    answer = query([img1], question)
    row['response'] = answer
    return row
    # breakpoint()
import pandas as pd
import os
from tqdm.cli import tqdm
import json
data = pd.read_csv("benchmark/gen.csv")  # Assume data.csv has columns: image1_path, image2_path, question
all_qa = []
data = data

print("HH0")
res = process_row(data.iloc[0])

from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import io
print("HH")
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(process_row, row) for idx, row in tqdm(data.iterrows())]
    for future in tqdm(as_completed(futures)):
        try:
            result = future.result()
            all_qa.append(result)
        except Exception as e:
            print(e)
            continue
        
        
df = pd.DataFrame(all_qa)
df.to_csv(f"outputs/test_result_{out_name}_gen_full.csv", index=False)
