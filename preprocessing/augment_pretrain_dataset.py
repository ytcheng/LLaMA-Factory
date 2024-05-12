from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from bs4 import BeautifulSoup
import re
import time
import torch
# count = 0
# def augment(content):
#     # count = count + 1
#     return content


model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-32B-Chat-GPTQ-Int4",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-32B-Chat-GPTQ-Int4")
prompt = "用通顺流畅的语言重新表达下文内容。务必不要用类似\"xxx: xxxxxxx\"或”xxx: \n\" 这种句式。要覆盖原文的全部内容，不要遗漏。\n\n"

def augment(content):
    device = "cuda" # the device to load the model onto
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt + content}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=2048
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    torch.cuda.empty_cache()
    get_gpu_memory_usage()
    print(response)
    return response

def get_gpu_memory_usage():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("CUDA device detected.")
        print("Device name:", torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0)/1024**3,1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0)/1024**3,1), "GB")

pattern = re.compile(r'\[img\][^\[^\]]*\[/img\]')
def remove_html_tags(content):
    soup = BeautifulSoup(content, "html.parser")
    clean_text = soup.get_text()
    clean_text = clean_text.replace('\u3000', '')
    clean_text = clean_text.replace('\xa0','')
    clean_text = re.sub(pattern, ' ', clean_text)
    return clean_text
def time_format(timestamp):
    timeArray = time.localtime(timestamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime


# 处理论坛贴子
def forum_augment(example):
    if example["status"] != 1 or len(example["content"]) < 500:
        example["augmented_contents"] = []
        return example

    times = 1
    if example["subtype"] == 5:
        times = 5
    augmented_contents = []
    for i in range(times):
        content = augment(example["clean_text"])
        augmented_contents.append(content)
    example["augmented_contents"] = augmented_contents
    return example

forum_dataset = load_dataset("ytcheng/sm_forum")
forum_dataset = forum_dataset.map(lambda example: {"clean_text": remove_html_tags(example["content"])})
forum_dataset = forum_dataset.map(forum_augment)
print(forum_dataset)
print(forum_dataset["train"][0])
forum_dataset.push_to_hub("ytcheng/sm_forum")


# 处理攻略站文章
def strategy_augment(example):
    times = 5
    augmented_contents = []
    for i in range(times):
        content = augment(example["clean_text"])
        augmented_contents.append(content)
    example["augmented_contents"] = augmented_contents
    return example
strategy_dataset = load_dataset("ytcheng/sm_strategy")
strategy_dataset = strategy_dataset.map(lambda example: {"clean_text": remove_html_tags(example["content"])})
strategy_dataset = strategy_dataset.map(strategy_augment)
print(strategy_dataset)
print(strategy_dataset["train"][0])
strategy_dataset.push_to_hub("ytcheng/sm_strategy")

# 处理新闻文章
def news_augment(example):
    if example["content"] == "":
        example["augmented_contents"] = []
        return example

    times = 1
    if example["type"] == 5:
        times = 5
    augmented_contents = []
    for i in range(times):
        content = augment(example["clean_text"])
        augmented_contents.append(content)
    example["augmented_contents"] = augmented_contents
    return example
article_dataset = load_dataset("ytcheng/sm_news")
article_dataset = article_dataset.map(lambda example: {"clean_text": remove_html_tags(example["content"])})
article_dataset = article_dataset.map(news_augment)
print(article_dataset)
print(article_dataset["train"][0])
article_dataset.push_to_hub("ytcheng/sm_news")

print("c:")
print(c)