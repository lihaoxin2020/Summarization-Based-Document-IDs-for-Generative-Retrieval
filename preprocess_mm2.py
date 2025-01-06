# %%
import json
from datasets import Dataset
from transformers import AutoTokenizer

# %%
train_d2k = json.load(open("mm2-data/mm2-100k/mm2-100k-train-d2k.json"))
valid_d2k = json.load(open("mm2-data/mm2-100k/mm2-100k-val-d2k.json"))
test_d2k = json.load(open("mm2-data/mm2-100k/mm2-100k-test-d2k.json"))
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")
# %%
semantic_examples = []
summ_examples = []
for _, values in train_d2k.items():
    for query in values['queries'] + values['synthetic_queries']:
    # for query in values['queries']:
        query = query.strip()
        prompt = "Query: " + query + " Keys:"
        prompt_len = len(tokenizer(prompt)['input_ids'])
        semantic_example = tokenizer(prompt + " " + values['semantic_id'] + tokenizer.eos_token)
        # semantic_example['labels'] = semantic_example['input_ids'].copy()
        semantic_example['labels'] = [-100] * prompt_len + semantic_example['input_ids'][prompt_len:]
        summ_example = tokenizer("Query: " + query + " Keys: " + values['key'] + tokenizer.eos_token)
        # summ_example['labels'] = summ_example['input_ids'].copy()
        summ_example['labels'] = [-100] * prompt_len + summ_example['input_ids'][prompt_len:]
        semantic_examples.append(semantic_example)
        summ_examples.append(summ_example)
    for doc in values['doc']:
        doc = doc.strip()
        prompt = "Text: " + doc + " Keys:"
        prompt_len = len(tokenizer(prompt)['input_ids'])
        semantic_example = tokenizer("Text: " + doc + " Keys: " + values['semantic_id'] + tokenizer.eos_token)
        # semantic_example['labels'] = semantic_example['input_ids'].copy()
        semantic_example['labels'] = [-100] * prompt_len + semantic_example['input_ids'][prompt_len:]
        summ_example = tokenizer("Text: " + doc + " Keys: " + values['key'] + tokenizer.eos_token)
        # summ_example['labels'] = summ_example['input_ids'].copy()
        summ_example['labels'] = [-100] * prompt_len + summ_example['input_ids'][prompt_len:]
        semantic_examples.append(semantic_example)
        summ_examples.append(summ_example)

for _, values in valid_d2k.items():
    for query in values['synthetic_queries']:
        query = query.strip()
        prompt = "Query: " + query + "\nKeys:"
        prompt_len = len(tokenizer(prompt)['input_ids'])
        semantic_example = tokenizer("Query: " + query + "\nKeys: " + values['semantic_id'] + tokenizer.eos_token)
        # semantic_example['labels'] = semantic_example['input_ids'].copy()
        semantic_example['labels'] = [-100] * prompt_len + semantic_example['input_ids'][prompt_len:]
        summ_example = tokenizer("Query: " + query + "\nKeys: " + values['key'] + tokenizer.eos_token)
        # summ_example['labels'] = summ_example['input_ids'].copy()
        summ_example['labels'] = [-100] * prompt_len + summ_example['input_ids'][prompt_len:]
        semantic_examples.append(semantic_example)
        summ_examples.append(summ_example)
    for doc in values['doc']:
        doc = doc.strip()
        prompt = "Text: " + doc + " Keys:"
        prompt_len = len(tokenizer(prompt)['input_ids'])
        semantic_example = tokenizer("Text: " + doc + " Keys: " + values['semantic_id'] + tokenizer.eos_token)
        # semantic_example['labels'] = semantic_example['input_ids'].copy()
        semantic_example['labels'] = [-100] * prompt_len + semantic_example['input_ids'][prompt_len:]
        summ_example = tokenizer("Text: " + doc + " Keys: " + values['key'] + tokenizer.eos_token)
        # summ_example['labels'] = summ_example['input_ids'].copy()
        summ_example['labels'] = [-100] * prompt_len + summ_example['input_ids'][prompt_len:]
        semantic_examples.append(semantic_example)
        summ_examples.append(summ_example)

for _, values in test_d2k.items():
    for query in values['synthetic_queries']:
        query = query.strip()
        prompt = "Query: " + query + "\nKeys:"
        prompt_len = len(tokenizer(prompt)['input_ids'])
        semantic_example = tokenizer("Query: " + query + "\nKeys: " + values['semantic_id'] + tokenizer.eos_token)
        # semantic_example['labels'] = semantic_example['input_ids'].copy()
        semantic_example['labels'] = [-100] * prompt_len + semantic_example['input_ids'][prompt_len:]
        summ_example = tokenizer("Query: " + query + "\nKeys: " + values['key'] + tokenizer.eos_token)
        # summ_example['labels'] = summ_example['input_ids'].copy()
        summ_example['labels'] = [-100] * prompt_len + summ_example['input_ids'][prompt_len:]
        semantic_examples.append(semantic_example)
        summ_examples.append(summ_example)
    for doc in values['doc']:
        doc = doc.strip()
        prompt = "Text: " + doc + " Keys:"
        prompt_len = len(tokenizer(prompt)['input_ids'])
        semantic_example = tokenizer("Text: " + doc + " Keys: " + values['semantic_id'] + tokenizer.eos_token)
        # semantic_example['labels'] = semantic_example['input_ids'].copy()
        semantic_example['labels'] = [-100] * prompt_len + semantic_example['input_ids'][prompt_len:]
        summ_example = tokenizer("Text: " + doc + " Keys: " + values['key'] + tokenizer.eos_token)
        # summ_example['labels'] = summ_example['input_ids'].copy()
        summ_example['labels'] = [-100] * prompt_len + summ_example['input_ids'][prompt_len:]
        semantic_examples.append(semantic_example)
        summ_examples.append(summ_example)


# %%
train_1k_sem = Dataset.from_list(semantic_examples)
train_1k_sum = Dataset.from_list(summ_examples)
# %%
# train_1k_sem.save_to_disk("mm2-data/mm2-1k-train-cluster")
# train_1k_sum.save_to_disk("mm2-data/mm2-1k-train-llm")
# %%
semantic_examples = []
summ_examples = []
for _, values in valid_d2k.items():
    for query in values['queries']:
        query = query.strip()
        semantic_example = tokenizer("Query: " + query + " Keys:")
        semantic_example['labels'] = tokenizer("Query: " + query + " Keys: " + values['semantic_id'] + tokenizer.eos_token)['input_ids']
        summ_example = tokenizer("Query: " + query + " Keys:")
        summ_example['labels'] = tokenizer("Query: " + query + " Keys: " + values['key'] + tokenizer.eos_token)['input_ids']
        semantic_examples.append(semantic_example)
        summ_examples.append(summ_example)
# %%
valid_1k_sem = Dataset.from_list(semantic_examples)
valid_1k_sum = Dataset.from_list(summ_examples)
# %%
semantic_examples = []
summ_examples = []
for _, values in test_d2k.items():
    for query in values['queries']:
        query = query.strip()
        semantic_example = tokenizer("Query: " + query + " Keys:")
        semantic_example['labels'] = tokenizer("Query: " + query + " Keys: " + values['semantic_id'] + tokenizer.eos_token)['input_ids']
        summ_example = tokenizer("Query: " + query + " Keys:")
        summ_example['labels'] = tokenizer("Query: " + query + " Keys: " + values['key'] + tokenizer.eos_token)['input_ids']
        semantic_examples.append(semantic_example)
        summ_examples.append(summ_example)
# %%
test_1k_sem = Dataset.from_list(semantic_examples)
test_1k_sum = Dataset.from_list(summ_examples)
# %%
from datasets import DatasetDict

semantic_id_ds = DatasetDict({'train': train_1k_sem, 'validation': valid_1k_sem, 'test': test_1k_sem})
generate_id_ds = DatasetDict({'train': train_1k_sum, 'validation': valid_1k_sum, 'test': test_1k_sum})
semantic_id_ds.save_to_disk('mm2-data/mm2-test/mm2-100k-cluster-only-keys-w-text')
generate_id_ds.save_to_disk('mm2-data/mm2-test/mm2-100k-llm-only-keys-w-text')
# %%
