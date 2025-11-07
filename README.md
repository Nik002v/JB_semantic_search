# Code Search Engine Project

## Description
This project implements a code search engine using dense embeddings and a Weaviate vector database. 
All code, including model training, fine-tuning, and evaluation on the CoSQA dataset, is contained in the notebook.

## Repository Structure

- `notebooks/cosqa_evaluation.ipynb` - Complete solution, including:
  - Dataset loading and preprocessing
  - Model embedding and fine-tuning
  - Vector database initialization and indexing
  - Evaluation (Recall@k, MRR@k, NDCG@k)
  - Plots and results summary

## Setup & Running

You can run the notebook directly in **Google Colab**:

1. Open Colab: [https://colab.research.google.com/](https://colab.research.google.com/)
2. Upload `cosqa_evaluation.ipynb` or open it from GitHub.
3. Make sure to **enable GPU**:
   - Runtime → Change runtime type → Hardware accelerator → GPU (T4 is free)
4. Run all cells.  
   **Note:** Running on CPU is much slower, so GPU is recommended for embedding computation and fine-tuning.

## Notes

- `Config.func_name_embedding` controls whether embeddings are computed from function names or full function bodies.
- `Config.vector_index_config` contains Weaviate HNSW hyperparameters affecting search metrics.
- Everything runs in the notebook, no separate `requirements.txt` is required.
