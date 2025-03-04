# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='allenai/longformer-base-4096', help='model to run')
parser.add_argument('--mode', type=str, default='cluster', help='attn or cluster')
parser.add_argument('--dataset', type=str, default=None, help='dataset load_from_disk')
parser.add_argument('--js_dataset', type=str, default=None, help='dataset load_from_disk')
parser.add_argument('--data_files', type=str, default=None, help='data files load_dataset')
parser.add_argument('--output_dir', type=str, default='nq-max-avg_embeddings_resume.pt')
parser.add_argument('--resume_from', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--shards', type=int, default=1)
parser.add_argument('--shard_id', type=int, default=0)

class SimpleDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        ex = self.dataset[index]
        doc = ex['text']
        return doc

def main():
    args = parser.parse_args()
    # ds = args.dataset if args.dataset is not None else args.data_files
    model_name = args.model_name_or_path
    # model_name = 'bert-base-cased'
    shards = args.shards
    shard_id = args.shard_id
    device = f'cuda:{shard_id}' if torch.cuda.is_available() else 'cpu'
    batch_size = args.batch_size
    mode = args.mode
    resume_from = args.resume_from
    output_dir = args.output_dir

    if args.dataset is not None:
        ds = load_dataset(
            'json', 
            data_files=args.dataset,
            cache_dir='.cache'
        )
        ds_train = ds['train']
    elif args.js_dataset is not None:
        ds = json.load(open(args.js_dataset))
    else:
        ds_train = load_from_disk(ds)

    # %%

    if shards != 1:
        shard_size = len(ds_train) // shards
        if shard_id == shards - 1:
            ds_train = ds_train.select(range(shard_size * shard_id, len(ds_train)))
        else:
            ds_train = ds_train.select(range(shard_size * shard_id, shard_size * (shard_id + 1)))


    # %%
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        cache_dir='../.cache'
    )
    model = AutoModel.from_pretrained(
        model_name, 
        cache_dir='../.cache',
        # torch_dtype=torch.float16
        torch_dtype='auto', 
        add_pooling_layer=False, 
    ).to(device)
    model.eval()
    # head_dim = model.config.hidden_size // model.config.num_attention_heads
    # num_heads = model.config.num_attention_heads
    # model_length = tokenizer.model_max_length

    outputs = {}
    if resume_from is not None:
        with open(resume_from, 'r') as f:
            outputs = json.load(f)
            ds_train = ds_train[len(outputs):]

    dataloader = DataLoader(
        ds_train,
        shuffle=False,
        batch_size=batch_size
    )

    # %%
    for i, text in enumerate(tqdm(dataloader)):
        for doc in text['document_text']:
            if doc in outputs:
                continue
        text = text['document_text']
        inputs = tokenizer(text, truncation=True, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            output = model(
                **inputs, 
                output_attentions=True
            )
            if mode == 'cluster':
                sum_pooled = torch.sum(output.last_hidden_state * inputs['attention_mask'].unsqueeze(-1), dim=1)
                weights = torch.sum(inputs['attention_mask'], dim=-1)
                avg_pooled = (sum_pooled / weights.unsqueeze(-1)).tolist()
                for j, doc in enumerate(text):
                    outputs[doc] = avg_pooled[j]
            elif mode == 'attn':
                attn_full = output.attentions[-1].cpu()
                attn_weights = attn_full.sum(-2) / inputs['attention_mask'].sum(dim=-1).cpu().unsqueeze(-1).unsqueeze(-1)
                token_index = torch.topk(attn_weights, 15)[1]
                tokens = inputs['input_ids'].cpu()[token_index]

        if i % 1000 == 0:
            torch.save(outputs, output_dir)

    torch.save(outputs, output_dir)

        
if __name__ == "__main__":
    main()

# %%
