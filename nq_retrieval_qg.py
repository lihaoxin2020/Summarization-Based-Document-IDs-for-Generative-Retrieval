# %%
from datasets import load_dataset, load_from_disk
import json
from transformers import AutoTokenizer

# %%
# ds_max = load_from_disk("nq-data/nq-max/nq-max-train")
# text2key = {ex['document_text']: ex['keyphrases'] for ex in ds_max}
text2key = json.load(open("nq-data/nq-100k-corpus.json"))

# %%
# doc_set = set(ds_max['document_text'])

# %%
ds_100k = load_dataset(
    'json',
    data_files="nq-data/nq-100k-train.txt",
    cache_dir=".cache"
)['train']
ds_valid = load_dataset(
    'json',
    data_files="nq-data/nq-true-dev.txt",
    cache_dir=".cache"
)['train']

# %%
# corpus = {}
# for ex in ds_100k:
#     text = ex['document_text']
#     corpus[text] = {'key': ex['keyphrases']}
# train_corpus = {}
# for ex in ds_100k:
#     text = ex['document_text']
#     train_corpus[text] = text2key[text]

# %%
tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-160m-deduped",
    cache_dir="../.cache",
    use_fast=True,
    model_max_length=512
)

# %%
# qrels_train = qrels['train']
train_ds = []
for ex in ds_100k:
    query = ex['query'].strip()
    text = ex['document_text']
    key = text2key[text].strip()
    prompt = "Query: " + query + "\nKeys:"
    prompt_len = len(tokenizer(prompt)['input_ids'])
    example = tokenizer(prompt + " " + key + tokenizer.eos_token)
    example['labels'] = [-100] * prompt_len + example['input_ids'][prompt_len:]
    train_ds.append(example)

# %%
train_synth_queries = json.load(open("nq-data/nq-max-queries.json"))
valid_synth_queries = json.load(open("nq-data/nq-valid-queries.json"))
# %%
synth_examples = []
doc_set = set()
for ex in ds_100k:
    _id = ex['id']
    queries = train_synth_queries[str(_id)]
    text = ex['document_text']
    if text not in doc_set:
        doc_set.add(text)
        key = text2key[text].strip()
        for query in queries:
            query = query.strip()
            prompt = "Query: " + query + "\nKeys:"
            prompt_len = len(tokenizer(prompt)['input_ids'])
            example = tokenizer(prompt + " " + key + tokenizer.eos_token)
            example['labels'] = [-100] * prompt_len + example['input_ids'][prompt_len:]
            synth_examples.append(example)
    
# %%
train_ds.extend(synth_examples)
# %%
valid_text2key = json.load(open("nq-data/nq-true-dev-llm.json"))
# %%
synth_valid_examples = []
doc_set = set()
for ex in ds_valid:
    _id = ex['id']
    queries = valid_synth_queries[str(_id)]
    text = ex['document_text']
    if text not in doc_set:
        doc_set.add(text)
        try:
            key = text2key[text].strip()
        except:
            key = valid_text2key[text].strip()
        for query in queries:
            query = query.strip()
            prompt = "Query: " + query + "\nKeys:"
            prompt_len = len(tokenizer(prompt)['input_ids'])
            example = tokenizer(prompt + " " + key + tokenizer.eos_token)
            example['labels'] = [-100] * prompt_len + example['input_ids'][prompt_len:]
            synth_valid_examples.append(example)


# %%
valid_ds = []
for ex in ds_valid:
    _id = ex['id']
    query = ex['query'].strip()
    text = ex['document_text']
    try:
        key = text2key[text].strip()
    except:
        key = valid_text2key[text].strip()
    prompt = "Query: " + query + "\nKeys:"
    example = tokenizer(prompt)
    prompt_len = len(tokenizer(prompt)['input_ids'])
    example['labels'] = [tokenizer(prompt + " " + key + tokenizer.eos_token)['input_ids']]
    assert example['input_ids'] == example['labels'][0][:prompt_len]
    valid_ds.append(example)

# %%
from datasets import Dataset, DatasetDict

ds_100k = Dataset.from_list(train_ds + synth_examples + synth_valid_examples)
val = Dataset.from_list(valid_ds)
ds = DatasetDict({'train': ds_100k, 'validation': val})

# %%
ds.save_to_disk("nq-data/retrieval-qg-100-llm")

# %%
# for ex in valid_text2key.items():
#     text = ex[0]
#     if text not in text2key:
#         text2key[text] = ex[1]

# %%
# output_list = []
# for ex in text2key.items():
#     output_list.append({'text': ex[0], 'key': ex[1]})

# # %%
# with open("nq-data/corpus-100k-llm.json", 'w') as f:
#     json.dump(output_list, f)

# %%
