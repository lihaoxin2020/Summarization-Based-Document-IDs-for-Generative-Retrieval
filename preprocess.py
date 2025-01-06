# %%
import datasets
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import json
from tqdm import tqdm

# %%

# for Pythia hyperparameters see https://arxiv.org/pdf/2304.01373.pdf
# initial LR for 2.8B model: 0.00016 with a min LR of 0.000016
# 2048 tokens seq len, 1024 samples per batch

device = "cuda" if torch.cuda.is_available() else 'cpu'
CACHE_DIR = '/data/keung/cache/retrieval'
dataset = 'nq-data/nq-10k-train.txt'
base_model = 'EleutherAI/pythia-2.8b-deduped'
# model = GPTNeoXForCausalLM.from_pretrained(
#     base_model,
#     torch_dtype=torch.float32,
#     load_in_8bit=False,
#     device_map='auto',
#     cache_dir=CACHE_DIR
# )

tokenizer = AutoTokenizer.from_pretrained(base_model)

# load the datasets
nq_train_data = json.load(open(dataset))
# nq_dev_data = json.load(open('./nq-dev.txt'))
# wiki_train_data = json.load(open('./min_passages.txt'))


#####################
# %%
# keywords2txt = {}
# for txt in wiki_train_data:
#     if txt['id'] in keywords2txt:
#         assert keywords2txt[txt['id']] == txt['text']
#     else:
#         keywords2txt[txt['id']] = txt['text']

# # %%
# nq_train_ids = set()
# for ex in nq_train_data:
#     nq_train_ids.add(ex['id'])
#     ex['text'] = keywords2txt[ex['id']]

# # for ex in nq_dev_data:
# #     # nq_train_ids.add(ex['id'])
# #     ex['text'] = keywords2txt[ex['id']]

# # # %%
# for ex in wiki_train_data:
#     if ex['id'] not in nq_train_ids:
#         nq_train_data.append({'keywords': ex['keywords'], 'text': ex['text']})

# train_dataset = nq_train_data
#####################

# # %%
# with open("train_test.txt", 'w') as fp:
#     json.dump(nq_train_data, fp)
    
# %%
def create_text(row, mode="train"):
    keywords = ' '.join(row['keywords'])
    # if 'question' in row:
    #     question = row['question']
    #     text = row['text'] if 'text' in row else None
    #     if mode == "eval":
    #         return (f'\nQuestion: {question} Keywords:', f'\nQuestion: {question} Keywords: {keywords}\n')
    #     return (f'\nQuestion: {question} Keywords: {keywords} Passage: {text}\n' 
    #             if text is not None 
    #             else f'\nQuestion: {question} Keywords: {keywords}\n')
    # else:
    #     text = row['text']
    #     return f'\nKeywords: {keywords} Passage: {text}\n'
    if 'text' in row:
        text = row['text']
        # text_lst = text.split()
        # text = ' '.join(text_lst[:32])
        return f'\nKeywords: {keywords} Passage: {text}'
    elif 'question' in row:
        text = row['question']
        if mode == "eval":
            return f'\nQuestion: {text} Keywords:', f'\nQuestion: {text} Keywords: {keywords}\n'
        return f'\nQuestion: {text} Keywords: {keywords}\n'
    else:
        raise("example keywords not found!")
   
# %%
max_length = 512
tokenized_dataset = []

for ex in tqdm(train_dataset):
    text = create_text(ex)
    tokenized = tokenizer(text)
    if len(tokenized['input_ids']) > max_length:
        print(text + ' is too long!')
    tokenized["labels"] = tokenized["input_ids"].copy()
    tokenized_dataset.append(tokenized)

dev_dataset = []
for ex in tqdm(nq_dev_data):
    text, label = create_text(ex, "eval")
    tokenized = tokenizer(text)
    tokenized["labels"] = tokenizer(label)["input_ids"]
    if len(tokenized['labels']) > max_length:
        print(text + ' is too long!')
    dev_dataset.append(tokenized)

# %%
tokenized_dataset = datasets.Dataset.from_list(tokenized_dataset)
dev_dataset = datasets.Dataset.from_list(dev_dataset)

# %%
# from sklearn.model_selection import train_test_split
#
# ready_train, ready_test = train_test_split(tokenized_dataset, test_size=0.01, random_state=42)
ready_dataset = datasets.DatasetDict({'train': tokenized_dataset, 'validation': dev_dataset})

# %%
ready_dataset.save_to_disk("./preprocessed_min_6.18")

# %%
