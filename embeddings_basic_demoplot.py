# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 12:13:15 2025

@author: JaishriRai
"""

from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

words = ['refund', 'money back', 'laptop', 'computer',
         'return policy', 'reimbursement', 'pc', 'notebook']

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(words, normalize_embeddings=True)  # cosine-friendly

tsne = TSNE(n_components=2, perplexity=5, random_state=42)  # a bit higher with more points
emb2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(emb2d[:, 0], emb2d[:, 1])
for i, w in enumerate(words):
    plt.annotate(w, (emb2d[i, 0], emb2d[i, 1]))
plt.title('t-SNE of Sentence Embeddings (unit-normalized)')
plt.show()