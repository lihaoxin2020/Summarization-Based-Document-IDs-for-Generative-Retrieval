# %%
import json
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import numpy as np

np.random.seed(0)
# model_name = "EleutherAI/pythia-160m-deduped"
model_name = "t5-base"

# %%
qrels = load_from_disk("beir_msmarco/beir_msmarco-qrels-100k")
synth_q = json.load(open("beir_msmarco/beir_msmarco-corpus-100k-queries.json"))
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir="../.cache",
    use_fast=True
)

# %%
queries = load_dataset(
    "BeIR/msmarco",
    "queries",
    cache_dir="../.cache"
)
corpus = json.load(open("beir_msmarco/beir_msmarco-corpus-100k-ckpt.json"))
corpus_id2key = {ex['_id']: ex['key'] for ex in corpus}
queries_id2q = {ex['_id']: ex['text'] for ex in queries['queries']}
# semantic_ids = json.load(open("beir_msmarco/beir_msmarco-corpus-100k-semantic_ids.json"))
corpus_id2text = {ex['_id']: ex['text'] for ex in corpus}

# %%
# qrels_train = qrels['train'].select(np.random.choice(len(qrels['train']), 1000, replace=False))
# qrels_train.save_to_disk("beir_msmarco/retrieval-qg-1k-qrels")

# %%
qrels_train = load_from_disk("beir_msmarco/retrieval-qg-10k-qrels")
# qrels_train = qrels['train']
train_ds = []
for ex in qrels_train:
    query_id = ex['query-id']
    corpus_id = ex['corpus-id']
    # queries = synth_q[str(corpus_id)] + [queries_id2q[str(query_id)]]
    queries = [queries_id2q[str(query_id)]]
    key = corpus_id2key[corpus_id].strip()
    text = corpus_id2text[corpus_id]
    # key = semantic_ids[text]
    for query in queries:
        query = query.strip()
        # prompt = "Query: " + query + "\nKeys:"
        prompt = "Query: " + query
        # prompt_len = len(tokenizer(prompt)['input_ids'])
        # example = tokenizer(prompt + " " + key + tokenizer.eos_token)
        # example['labels'] = [-100] * prompt_len + example['input_ids'][prompt_len:]
        example = tokenizer(prompt)
        example['labels'] = tokenizer(key)['input_ids']
        train_ds.append(example)
        # train_ds.append({
        #     "text": prompt,
        #     "labels": key
        # })

# %%
for ex in corpus:
    corpus_id = ex['_id']
    key = ex['key'].strip()
    queries = synth_q[str(corpus_id)]
    for query in queries:
        query = query.strip()
        # prompt = "Query: " + query + "\nKeys:"
        prompt = "Query: " + query
        # prompt_len = len(tokenizer(prompt)['input_ids'])
        # example = tokenizer(prompt + " " + key + tokenizer.eos_token)
        # example['labels'] = [-100] * prompt_len + example['input_ids'][prompt_len:]
        example = tokenizer(prompt)
        example['labels'] = tokenizer(key)['input_ids']
        train_ds.append(example)
        # train_ds.append({
        #     "text": prompt,
        #     "labels": key
        # })

# %%
qrels_valid = qrels['validation']
valid_dict = {}
for ex in qrels_valid:
    query_id = ex['query-id']
    corpus_id = ex['corpus-id']
    text = corpus_id2text[corpus_id]
    key = corpus_id2key[corpus_id].strip()
    query = queries_id2q[str(query_id)].strip()
    if query not in valid_dict:
        valid_dict[query] = [key]
    elif key not in valid_dict[query]:
        valid_dict[query].append(key)
        
# %%
valid_ds = []
for query, answers in valid_dict.items():
    # prompt = "Query: " + query + "\nKeys:"
    prompt = "Query: " + query
    # example = tokenizer(prompt)
    # example['labels'] = tokenizer([prompt + " " + key + tokenizer.eos_token for key in answers])['input_ids']
    example = tokenizer(prompt)
    example['labels'] = tokenizer([key for key in answers])['input_ids']
    valid_ds.append(example)
    # valid_ds.append({
    #     "text": prompt,
    #     "labels": answers[0]
    # })

# %%
from datasets import Dataset, DatasetDict
train_ds = Dataset.from_list(train_ds)
# train_ds = load_from_disk("beir_msmarco/t5-retrieval-qg-10k-llm")['train']
valid_ds = Dataset.from_list(valid_ds)
ds = DatasetDict({'train': train_ds, 'validation': valid_ds})
ds.save_to_disk("beir_msmarco/retrieval-qg-1k-llm-t5")

# %%
corpus_30 = []
for ex in corpus:
    corpus_id = ex['_id']
    text = ex['text']
    key_ids = tokenizer(text)['input_ids'][:30]
    key = tokenizer.decode(key_ids, skip_special_tokens=True)
    # assert key_ids == tokenizer(key)['input_ids']
    corpus_30.append({
        '_id': ex['_id'],
        'text': text,
        'key': key
    })

# %%
with open("beir_msmarco/beir_msmarco-corpus-100k-l30.json", 'w') as f:
    json.dump(corpus_30, f)

# %%
