from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import json
import re
from datasets import Dataset, DatasetDict
import hashlib

from bs4 import BeautifulSoup

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-32B-Chat-GPTQ-Int4",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-32B-Chat-GPTQ-Int4")
prompt = "从以下内容中，提取尽可能多的问题和答案，尽量覆盖所有的内容，问题和答案要包含必要的背景信息。以json格式返回，json格式为[{\"question\":\"xxxx\",\"answer\":\"xxx\"},xxxx]。\n\n"



def get_gpu_memory_usage():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("CUDA device detected.")
        print("Device name:", torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0)/1024**3,1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0)/1024**3,1), "GB")


pattern = re.compile(r'\[img\][^\[^\]]*\[/img\]')
def remove_html_tags(data):
    soup = BeautifulSoup(data["content"], "html.parser")
    clean_text = soup.get_text()
    clean_text = clean_text.replace('\u3000', '')
    clean_text = clean_text.replace('\xa0','')
    clean_text = re.sub(pattern, ' ', clean_text)

    data["content"] = clean_text
    return data


article_dataset = load_dataset("ytcheng/sm_news")
article_dataset = article_dataset.filter(lambda x: x["content"]!="").filter(lambda x: x["type"]==5)
columns_to_remove = list(article_dataset["train"].features)
columns_to_remove.remove("title")
columns_to_remove.remove("content")
article_dataset = article_dataset.filter(lambda x: x["content"]!="")
article_dataset = article_dataset.map(remove_html_tags, remove_columns=columns_to_remove)
print(article_dataset)
# print(article_dataset["train"][0])

strategy_dataset = load_dataset("ytcheng/sm_strategy")
columns_to_remove = list(strategy_dataset["train"].features)
columns_to_remove.remove("title")
columns_to_remove.remove("content")
strategy_dataset = strategy_dataset.map(remove_html_tags, remove_columns=columns_to_remove)
print(strategy_dataset)
# print(strategy_dataset["train"][0])

dataset = concatenate_datasets([strategy_dataset["train"], article_dataset["train"]])
# dataset = dataset.select(range(5))
# for i in range(1, 6):
dataset = dataset.map(gengrate, with_indices=True)
    # print(dataset[0])
print(dataset)
print(dataset[0])

generate_datasets = Dataset.from_list(dataset)
datasets_formatted_data = DatasetDict({"train":  generate_datasets})
datasets_formatted_data.save_to_disk("sm_question3")
datasets_formatted_data.push_to_hub("ytcheng/sm_question3")


def gengrate(sample, index):
    questions = []
    try:
        for idx in range(5):
            content = "标题:" + sample["title"] + "\n\n" +sample["content"]
            device = "cuda" # the device to load the model onto
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt + content}
            ]
            for question in questions:
                messages.extend([
                    {"role": "assistant", "content": question},
                    {"role": "user", "content": "再生成一些"}
                ])
            print(messages)
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
            questions.append(response)
            torch.cuda.empty_cache()
            get_gpu_memory_usage()
    except Exception as e:
        print("An error occurred:", e)
    print(questions)
    sample["questions"] = questions

    if index % 10 == 0:
        # 保存处理后的数据集
        dataset.save_to_disk(f"sm_question_{index}")


    return sample
