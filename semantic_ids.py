# %%
from sklearn.cluster import KMeans
import numpy as np
import torch
import json

output_dir = "nq-data/corpus-nq-100k-semantic_ids.json"

# %%
def generate_ids(X1_N, ids, k=10, c=30):
    def Cluster(X1_N, k):
        # Implement your clustering algorithm here
        # For example, you can use k-means clustering from scikit-learn:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X1_N)
        return kmeans.labels_

    def elementwiseStrConcat(list1, list2):
        # Concatenate the elements of two lists element-wise as strings
        return [str(item1) + str(item2) for item1, item2 in zip(list1, list2)]


    C1_10 = Cluster(X1_N, k)
    J = []
    ids_out = []

    for i in range(k):
        J_current = [i] * np.sum(C1_10 == i)
        id_current = ids[C1_10 == i]
        if np.sum(C1_10 == i) > c:
            J_rest, id_current = generate_ids(X1_N[C1_10 == i], id_current)
        else:
            J_rest = list(range(np.sum(C1_10 == i)))

        J_cluster = elementwiseStrConcat(J_current, J_rest)
        J.extend(J_cluster)
        ids_out.extend(id_current)

    # J = reorderToOriginal(J, ids_out)
    return J, ids_out

# %%
# valid_docs = torch.load("mm2-data/mm2-val/mm2-val-embeddings.pt")
# train_docs = torch.load("mm2-data/mm2-10k/mm2-10k-train-embeddings.pt")
# test_docs = torch.load("mm2-data/mm2-test/mm2-test-embeddings.pt")

# Xs = list(valid_docs.values()) + list(train_docs.values()) + list(test_docs.values())
# docs = list(valid_docs.keys()) + list(train_docs.keys()) + list(test_docs.keys())
# corpus = torch.load("nq-data/nq-max-avg_embeddings.pt")
corpus0 = torch.load("nq-data/nq-100k-avg_embeddings0.pt")
corpus1 = torch.load("nq-data/nq-100k-avg_embeddings1.pt")
corpus2 = torch.load("nq-data/nq-100k-avg_embeddings2.pt")
corpus3 = torch.load("nq-data/nq-100k-avg_embeddings3.pt")
# %%
dev_corpus = torch.load("nq-data/nq-dev-avg_embeddings.pt")
valid_corpus = torch.load("nq-data/nq-valid-avg_embeddings.pt")
corpus = corpus0 | corpus1 | corpus2 | corpus3 | dev_corpus | valid_corpus 
# %%
Xs = list(corpus.values())
docs = list(corpus.keys())
Xs = np.array(Xs)
ids = np.arange(len(Xs))
# %%
J, ids = generate_ids(Xs, ids)
# %%
outputs = [''] * len(J)
for i, id in enumerate(ids):
    doc_id = J[i]
    outputs[id] = doc_id

# %%
output_list = []
for doc, doc_id in zip(docs, outputs):
    output_list.append({
        'text': doc,
        'key': "doc" + doc_id
    })

# %%
with open(output_dir, 'w') as f:
    json.dump(output_list, f)

# %%
