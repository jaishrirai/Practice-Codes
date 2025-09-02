# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 14:00:05 2025

@author: JaishriRai
"""

from transformers import pipeline

# Load BERT pipeline
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# Try a masked sentence
result = fill_mask("Transformers are [MASK] for language understanding.")
result_1 = fill_mask("life is very [MASK]. What you plan might not work well but definitly if you are open and flexible you will find a way through. ")

for r in result:
    print(f"{r['sequence']} (prob={r['score']:.4f})")

for r in result_1:
    print(f"{r['sequence']} (prob={r['score']:.4f})")
