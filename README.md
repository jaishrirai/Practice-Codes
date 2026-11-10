# Practice Codes

Small hands-on experiments from my daily AI and ML practice. Each script runs on its own and uses public models and toy data only.

| Script | What it shows |
|---|---|
| `basic_model_performance_metrics.py` | Train a logistic regression on the iris data and read precision, recall, F1 and the confusion matrix |
| `embeddings_basic_demoplot.py` | Turn words into sentence embeddings and plot them in 2D with t-SNE to see similar meanings cluster |
| `bert_fillblank_demo.py` | Fill-in-the-blank with BERT: which words does the model think fit a masked slot |
| `gpt_nextwordgen_demo.py` | Next word generation with GPT-2 from a starter prompt |
| `mini_transformer_attention_demo.py` | Pull attention weights out of BERT for one sentence |

## Run

```bash
pip install transformers torch sentence-transformers scikit-learn matplotlib
python embeddings_basic_demoplot.py
```

Models download from Hugging Face on first run.
