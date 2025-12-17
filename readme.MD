# MobileWorldBench: Towards Semantic World Modeling For Mobile Agents

[Arxiv](https://arxiv.org/abs/2512.14014)

### Benchmark

Please download images from [Huggingface](https://huggingface.co/datasets/jacklishufan/mobileworldbench/tree/main). The questions and prompts can be found in folder `benchmark`

To run the qa tasks,  use the follwong script

```
python scripts/eval_qa.py <model_config> 
```


To run the generation tasks, use the follwong script

```
# first get the response
python scripts/eval_gen.py <model_config> 
# then get scores via GPT judge
python scripts/score_gpt4o.py <path_to_model_outputs> 
```

Model config is a YAML file that defines OpenAI compatiable REST API endpoints (e.g. served via VLLM). We provide an example in `config` folder


### Finetuned Model

We release our finetuned model on [Huggingface](https://huggingface.co/jacklishufan/Qwen3-VL-8B-Instruct-MobileWorld-SFT)

### MobileWorld Training Data

Images can be found in Huggingface:

[AiTW images](https://huggingface.co/datasets/jacklishufan/aitw/tree/main)
[Android Control Images](https://huggingface.co/datasets/jacklishufan/android_control)

Annotations can be found 

[MobileWorld Training Data](https://huggingface.co/datasets/jacklishufan/MobileWorld)


Note that HF autoconversion is buggy. Please manually download the full data and untar it.

