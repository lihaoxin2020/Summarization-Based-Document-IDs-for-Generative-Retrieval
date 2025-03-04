# Summarization-Based Document IDs for Generative Retrieval with Language Models

**[The codebase is aged. Feel free to drop me an email if this is relevant to you!]**

This is the (probably outdated) codebase for [Summarization-Based Document IDs for Generative Retrieval with Language Models](https://arxiv.org/abs/2311.08593). The dataset is uploaded [here](https://huggingface.co/datasets/lihaoxin2020/abstractive-content-based-IDs).

Feel free to leave an issue or email if you have any question! 

## Running Steps:
- For document IDs, Abstractive Content-based ID (ACID) was uploaded to HuggingFace [datasets](https://huggingface.co/datasets/lihaoxin2020/abstractive-content-based-IDs). It semantic IDs are desired, generate doc embeddings with example code from `codes-attention-based.py`, and then generate ID strings with the example provided in `semantic_ids.py`.
- `generate_queries.py` was used for query augmentation given texts. 
- `preprocess.py` provides an example for tokenization after combining generated query-answer pairs with generic pairs. (Tokenized dataset is uploaded [here](https://huggingface.co/datasets/lihaoxin2020/abstractive-content-based-IDs/tree/main/nq-100-tokenized))
- Train a generative search model with `python run_clm_generative_retrieval.py run_clm_retrieval_generative_config.json`.
The `corpus_file` should be a json file containing unique document to ID mapping for constraint decoding, e.g. `[{"text": "...", "key": "..."}]`. An example file for NQ-100k is [here](https://drive.google.com/drive/folders/1R9dy5V-_5dFn7ZqHk3WmfK_sqipiKXNz?usp=drive_link).

**Note**: the constrait decoding implementation is outdated and very inefficient. I suggest to check out recent efforts such as SGLang for more efficient decoding and deployment.  


```
@misc{li2024summarizationbaseddocumentidsgenerative,
      title={Summarization-Based Document IDs for Generative Retrieval with Language Models}, 
      author={Haoxin Li and Daniel Cheng and Phillip Keung and Jungo Kasai and Noah A. Smith},
      year={2024},
      eprint={2311.08593},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2311.08593}, 
}
```
