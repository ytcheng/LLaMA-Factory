from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from bs4 import BeautifulSoup
import re
import time
import torch
import json
# count = 0
# def augment(content):
#     # count = count + 1
#     return content

models = []
device_count = torch.cuda.device_count()
for rank in range(device_count)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-32B-Chat-GPTQ-Int4",
        torch_dtype="auto",
        device_map={"": "cuda:{rank}"}
    )
    models.append(model)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-32B-Chat-GPTQ-Int4", padding_side='left')
prompt = "用通顺流畅的语言重新表达下文内容。务必不要用类似\"xxx: xxxxxxx\"或”xxx: \n\" 这种句式。要覆盖原文的全部内容，不要遗漏。\n\n"
# def augment(contents):
#     print("contents:")
#     print(json.dumps(contents, ensure_ascii=False))
#     return contents
def augment(contents, rank):
    device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
    model = models[rank]
    # model.to(device)
    # device = "cuda" # the device to load the model onto
    texts = []
    for content in contents:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt + content}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(text)
    tokenizer.pad_token = tokenizer.eos_token
    model_inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)
    try:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    except Exception as e:
        print("An error occurred:", e)
        response = [''] * len(contents)
        # response = ""
    
    get_gpu_memory_usage()
    torch.cuda.empty_cache()
    print("response:")
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
def forum_augment(examples, rank):
    texts = []
    for idx, clean_text in enumerate(examples["clean_text"]):
        text = "标题:" + examples["title"][idx] + "\n\n" + clean_text
        texts.append(text)
    
    augmented_contents = augment(texts, rank)
    augmented_contents_array = []
    for augmented_content in augmented_contents:
        augmented_contents_array.append([augmented_content])

   
    examples["augmented_contents"] = augmented_contents_array
    print("after new augment:")
    print(json.dumps(examples["augmented_contents"], ensure_ascii=False))
    return examples

def forum_stategy_augment(examples, rank):
    texts = []
    for idx, clean_text in enumerate(examples["clean_text"]):
        for i in range(5):
            text = "标题:" + examples["title"][idx] + "\n\n" + clean_text
            texts.append(text)
    augmented_contents = augment(texts, rank)

    augmented_contents_array = []
    for idx in range(len(examples["clean_text"])):
        augmented_contents_item_array = []
        for i in range(5):
            index = idx * (i + 1) + i
            print("index:"+str(index))
            augmented_contents_item_array.append(augmented_contents[index])
        augmented_contents_array.append(augmented_contents_item_array)

    examples["augmented_contents"] = augmented_contents_array
    print("after new augment:")
    print(json.dumps(examples["augmented_contents"], ensure_ascii=False))
    return examples

forum_dataset = load_dataset("ytcheng/sm_forum")
forum_dataset = forum_dataset.map(lambda example: {"clean_text": remove_html_tags(example["content"])})
forum_dataset = forum_dataset.filter(lambda x: len(x["clean_text"]) > 300 and x["status"] == 1)

forum_stategy_dataset = forum_dataset.filter(lambda x: x["subtype"] == 5)
forum_stategy_dataset = forum_stategy_dataset.map(forum_stategy_augment, batched=True, batch_size=2, num_proc=torch.cuda.device_count())
forum_stategy_dataset.push_to_hub("ytcheng/forum_stategy")
forum_artile_dataset = forum_dataset.filter(lambda x: x["subtype"] != 5)
forum_artile_dataset = forum_artile_dataset.map(forum_augment, batched=True, batch_size=5, num_proc=torch.cuda.device_count())
print(forum_artile_dataset)
print(forum_artile_dataset["train"][0])
forum_artile_dataset.push_to_hub("ytcheng/forum_article")


# 处理攻略站文章
def strategy_augment(example):
    times = 5
    augmented_contents = []
    texts = []
    for i in range(times):
        texts.append(example["clean_text"])
    augmented_contents = augment(texts)
    example["augmented_contents"] = augmented_contents
    return example
# strategy_dataset = load_dataset("ytcheng/sm_strategy")
# # strategy_dataset = strategy_dataset['train'].select(range(2))
# # strategy_dataset = DatasetDict({"train":  strategy_dataset})
# strategy_dataset = strategy_dataset.map(lambda example: {"clean_text": remove_html_tags(example["content"])})
# strategy_dataset = strategy_dataset.map(strategy_augment)
# print(strategy_dataset)
# print((json.dumps(strategy_dataset["train"][0], ensure_ascii=False)))
# strategy_dataset.save_to_disk("sm_strategy")
# strategy_dataset.push_to_hub("ytcheng/sm_strategy")

# 处理新闻文章
def news_augment(examples):
    texts = []
    for idx, clean_text in enumerate(examples["clean_text"]):
        text = "标题:" + examples["title"][idx] + "\n\n" + clean_text
        texts.append(text)
    
    augmented_contents = augment(texts)
    augmented_contents_array = []
    for augmented_content in augmented_contents:
        augmented_contents_array.append([augmented_content])

   
    examples["augmented_contents"] = augmented_contents_array
    print("after new augment:")
    print(json.dumps(examples["augmented_contents"], ensure_ascii=False))
    return examples

def news_stategy_augment(examples):
    texts = []
    for idx, clean_text in enumerate(examples["clean_text"]):
        for i in range(5):
            text = "标题:" + examples["title"][idx] + "\n\n" + clean_text
            texts.append(text)
    augmented_contents = augment(texts)

    augmented_contents_array = []
    for idx in range(len(examples["clean_text"])):
        augmented_contents_item_array = []
        for i in range(5):
            index = idx * (i + 1) + i
            print("index:"+str(index))
            augmented_contents_item_array.append(augmented_contents[index])
        augmented_contents_array.append(augmented_contents_item_array)

    examples["augmented_contents"] = augmented_contents_array
    print("after new augment:")
    print(json.dumps(examples["augmented_contents"], ensure_ascii=False))
    return examples

# news_dataset = load_dataset("ytcheng/sm_news")
# news_dataset = news_dataset.map(lambda example: {"clean_text": remove_html_tags(example["content"])})
# news_dataset = news_dataset.filter(lambda x: len(x["clean_text"]) > 50)

# article_dataset = news_dataset.filter(lambda x: x["type"] != 5)
# article_dataset = article_dataset['train'].select(range(10))
# article_dataset = DatasetDict({"train":  article_dataset})
# article_dataset = article_dataset.map(news_augment, batched=True, batch_size=10)
# article_dataset.save_to_disk("data/news_article")
# article_dataset.push_to_hub("ytcheng/news_article")

# news_stategy_dataset = news_dataset.filter(lambda x: x["type"] == 5)
# # news_stategy_dataset = news_stategy_dataset['train'].select(range(2))
# # news_stategy_dataset = DatasetDict({"train":  news_stategy_dataset})
# # print(news_stategy_dataset)
# news_stategy_dataset = news_stategy_dataset.map(news_stategy_augment, batched=True, batch_size=2)
# # print(news_stategy_dataset["train"][0])
# news_stategy_dataset.save_to_disk("data/news_stategy")
# news_stategy_dataset.push_to_hub("ytcheng/news_stategy")
