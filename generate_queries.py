# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
import json
import torch
from tqdm import tqdm

set_seed(42)
# %%
model_name_or_path = "doc2query/all-with_prefix-t5-base-v1"
data_disk = None
js_data = None
data_files = "nq-data/nq-dev.txt"
batch_size = 16
num_seq = 15
output_dir = 'nq-data/nq-dev-queries.json'
device = "cuda" if torch.cuda.is_available() else 'cpu'

# %%
def main():
    class SimpleDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            ex = self.dataset[index]
            return ex['_id'], ex['text']
        
    # %%
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, 
        cache_dir='.cache',
        use_fase=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
        cache_dir='.cache'
    ).to(device)
    # %%
    if data_disk is not None:
        dataset = load_from_disk(
            data_disk
        )
    elif js_data is not None:
        dataset = SimpleDataset(json.load(open(js_data)))
    else:
        dataset = load_dataset(
            "json", 
            data_files=data_files,
            cache_dir=".cache",
            split='train'
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
        # pin_memory=True
    )

    # %%
    prefix = "text2query: "
    outputs = {}
    for i, ex in enumerate(tqdm(dataloader)):
        inputs = [prefix + text for text in ex['document_text']]
        encoding = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt').to(device)
        generated = model.generate(
            **encoding, 
            max_new_tokens=64,
            # num_beams=5,
            temperature=1.2, 
            do_sample=True, 
            num_return_sequences=num_seq
        )
        generated_text = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for j, id in enumerate(ex["id"]):
            outputs[int(id)] = generated_text[j*num_seq : (j+1)*num_seq]
        
        if i % 10000 == 0:
            with open(output_dir, 'w') as f:
                json.dump(outputs, f)

    # %%
    with open(output_dir, 'w') as f:
        json.dump(outputs, f)


if __name__ == "__main__":
    main()