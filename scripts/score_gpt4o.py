# example call open AI API with GPT-4o on local images
from openai import OpenAI,AsyncOpenAI
import base64
import argparse
import yaml
parser = argparse.ArgumentParser()
parser.add_argument('output', type=str, help='Path to the config YAML file')
parser.add_argument('--image_root',default='mobileworldbench/gen_images', type=str, help='Root directory for images')
parser.add_argument('--config',default=None,type=str, help='Path to the config YAML file for judge model')
parser.add_argument('--limit',default=None,type=str, help='limit the number of eval samples, useful for debugging')

args = parser.parse_args()


if args.config is None:
    client = OpenAI()
    model = "gpt-4o-mini"
else:
    with open(args.config) as f:
        data = yaml.safe_load(f)
    # breakpoint()
    client = OpenAI(
        base_url=data['base_url'],
        api_key=data['api_key']  # vLLM doesn’t require authentication
    )   
    model = data['model']


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


PROMPT = '''
You are a judge to judge a VLMs's ability to understand the action on mobile device GUI. Given the current GUI screenshot and input action, the model will describe the changes that will occur on the next screen after the action is performed.\n The action is {action}.
You are provided with the model's response, an input image of the current state (first image), as well as ground truth next state image (second image).
You are also given a reference text that describes the actual changes that happened after performing the action.
Your task is to evaluate the model's response based on the following criteria:
1. Senmantic Correctness: Does the model's high level understanding matches the actucal changes occured? This criterion does not require you consider low level details that are hard to predict (e.g. news articles, produces after scroll)
2. Completeness: Does the model mention all relavent changes that are visible in the next state image?
3. Relevance: Are all the changes mentioned by the model relevant to the action performed?
For each criterion, assign a score from 1 to 5, where 1 is poor and 5 is excellent.
After evaluating all three criteria, provide an overall score out of 15, along with a brief justification for your scores.
The model's response is {response}
The reference changes are {changes}
You should focus on high-level senmantics and not too focused on small details
Format your evaluation as follows:
----Begin of response----
Accuracy: <score>
Completeness: <score>
Relevance: <score>
Overall Score: <total score>
----End of response----
use the exact format without any additional text.
'''

def extract_score_from_response(response_text):
    retry = 6
    while retry > 0:
        try:
            scores = {}
            lines = response_text.strip().split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key in ['Accuracy', 'Completeness', 'Relevance', 'Overall Score']:
                        scores[key] = int(value)
            return scores
        except Exception as e:
            print("Error parsing response, retrying...", e)
            retry -= 1
    return {}

#
import os
def process_row(row):
    row = dict(row)
    img1 = os.path.join(args.image_root,row['input_image'])
    img2 = os.path.join(args.image_root,row['output_image'])
    
    changes = row['changes']
    question = PROMPT.format(action=row['action'], response=row['response'],changes=changes)
    answer = query([img1,img2], question)
    scores = extract_score_from_response(answer)
    row.update(scores)
    # print('scores',answer,scores)
    # breakpoint()
    return row
    # breakpoint()
import pandas as pd
import os
from tqdm.cli import tqdm
import json
import sys
in_file = args.output
out_file = in_file.replace('.csv','_scored.csv')
data = pd.read_csv(in_file)  # Assume data.csv has columns: image1_path, image2_path, question
all_qa = []
data = data
process_row(data.iloc[0])
if args.limit is not None:
    data  = data[:args.limit]
print(len(data))
from concurrent.futures import ThreadPoolExecutor, as_completed
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(process_row, row) for idx, row in tqdm(data.iterrows())]
    for future in tqdm(as_completed(futures)):
        result = future.result()
        all_qa.append(result)
        df = pd.DataFrame(all_qa)
        r = df[['Accuracy', 'Completeness', 'Relevance', 'Overall Score']].mean()
        print(r)
        
df = pd.DataFrame(all_qa)

df.to_csv(out_file, index=False)

r = df[['Accuracy', 'Completeness', 'Relevance', 'Overall Score']].mean()
# 