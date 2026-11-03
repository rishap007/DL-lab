# DL lab

Small, standalone deep learning and machine learning examples.

## Word2Vec embeddings

`word2vec_embeddings.py` trains a small Word2Vec model on a three-sentence sample corpus, prints the embedding for `learning`, and lists similar words. A fixed seed makes the example reproducible.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python word2vec_embeddings.py
```
