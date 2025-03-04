# %%
import datasets
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

# %%
device = "cuda" if torch.cuda.is_available() else 'cpu'
CACHE_DIR = '/data/keung/cache/retrieval'
dataset = 'lihaoxin2020/abstractive-content-based-IDs'
base_model = 'EleutherAI/pythia-2.8b-deduped'

tokenizer = AutoTokenizer.from_pretrained(base_model)

# load the datasets
nq_train_data = datasets.load_dataset(dataset)['train']
nq_dev_data = datasets.load_dataset(dataset)['validation']
nq_test_data = datasets.load_dataset(dataset)['test']

#####################
# %%
def create_text(row, mode="train"):
    keywords = row['acid']
    # keywords = ' '.join(row['keywords'])
    if 'query' in row:
        text = row['query']
        if mode == "eval":
            return f'\nQuestion: {text} Keywords:', f'\nQuestion: {text} Keywords: {keywords}\n'
        return f'\nQuestion: {text} Keywords: {keywords}\n'
    else:
        raise("example keywords not found!")
   
# %%
max_length = 512
tokenized_dataset = []

for ex in tqdm(nq_train_data):
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

test_dataset = []
for ex in tqdm(nq_test_data):
    text, label = create_text(ex, "eval")
    tokenized = tokenizer(text)
    tokenized["labels"] = tokenizer(label)["input_ids"]
    if len(tokenized['labels']) > max_length:
        print(text + ' is too long!')
    test_dataset.append(tokenized)

# %%
tokenized_dataset = datasets.Dataset.from_list(tokenized_dataset)
dev_dataset = datasets.Dataset.from_list(dev_dataset)
test_dataset = datasets.Dataset.from_list(test_dataset)

# %%
ready_dataset = datasets.DatasetDict({'train': tokenized_dataset, 'validation': dev_dataset, 'test': test_dataset})

# %%
ready_dataset.save_to_disk("nq-data/retrieval-qg-100k-llm")

# %%
