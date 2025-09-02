# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 14:05:09 2025

@author: JaishriRai
"""

from transformers import pipeline, set_seed

# Load GPT pipeline
gen = pipeline("text-generation", model="gpt2")
set_seed(42)

# Try a starter prompt
result = gen(
    "Transformers changed NLP by",
    max_new_tokens=40,      # length of continuation
    do_sample=True,
    top_p=0.9,              # nucleus sampling
    temperature=0.8         # randomness control
)

print(result[0]['generated_text'])
