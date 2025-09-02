# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 16:55:56 2025

@author: JaishriRai
"""

# !pip install transformers torch --quiet  (if needed)
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name, output_attentions=True)
model = AutoModel.from_pretrained(model_name, config=config)

text = "Transformers use attention to focus on relevant words in context."
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)

emb = out.last_hidden_state            # [batch, seq_len, hidden]
cls_embedding = emb[0,0]               # [CLS] vector you can treat as sentence embedding (demo)
attn = out.attentions[-1][0]           # last layer attention: [heads, seq, seq]
print("CLS embed shape:", cls_embedding.shape)
print("Attention map shape (last layer):", attn.shape)
