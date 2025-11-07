# JB Semantic Search

This repository contains the implementation of a **semantic search system for code and documents**, including fine-tuning, vector storage, and evaluation metrics.  
The project can be run locally or on **Google Colab with a free T4 GPU** for faster execution.

> **Note:** The full Table of Contents is available inside the notebook for easier navigation.

---

## Table of Contents

1. **Libraries (`LIBS`)**  
   Lists all required Python libraries used in the project for data processing, embedding, and vector search.

2. **Configuration (`Config`)**  
   Contains configuration settings, including API keys, vector database parameters, and fine-tuning hyperparameters.

3. **Client (`CLIENT`)**  
   Code for interacting with the search engine or vector database, sending queries, and retrieving results.

4. **Document Schemas**  
   Defines the structure of documents and data objects used in the vector store.

5. **Search Engine**  
   Implements the semantic search logic, including indexing, embedding, and query handling.

6. **Metrics (`METRICS`)**  
   Contains evaluation metrics for measuring search quality, similarity, and retrieval performance.

7. **Loss Function Selection** *(discussion/text)*  
   Explanation of the different loss functions considered for fine-tuning the embedding model and rationale for selection.

8. **Fine-tuning**  
   Code and scripts for fine-tuning the embedding model on your dataset for improved semantic search accuracy.

9. **Data Loading (`DATA LOADING`)**  
   Handles loading, preprocessing, and batching of datasets for embedding or fine-tuning.

10. **Demos (`DEMOS`)**  
    Example notebooks and scripts demonstrating how to use the search system with sample queries.

11. **Main Execution (`MAIN`)**  
    Entry point of the project, orchestrating the workflow from data loading to search evaluation.

12. **Effect of Using Function Names Only** *(discussion/text)*  
    Analysis of how restricting input to only function names impacts embedding quality and search performance.

13. **Effect of Updating Vector Storage Hyperparameters** *(discussion/text)*  
    Discussion on how tuning the vector store parameters affects retrieval accuracy and performance.

---

## Setup & Running

You can run the notebook directly in **Google Colab**:

1. Open Colab: [https://colab.research.google.com/](https://colab.research.google.com/)
2. Upload `cosqa_evaluation.ipynb` or open it directly from GitHub.
3. Make sure to **enable GPU**:
   - Runtime → Change runtime type → Hardware accelerator → GPU (T4 is free)
4. Run all cells.  

**Note:** Running on CPU is much slower, so GPU is recommended for embedding computation and fine-tuning.

---

## Notes

- `Config.func_name_embedding` controls whether embeddings are computed from function names or full function bodies.  
- `Config.vector_index_config` contains Weaviate HNSW hyperparameters affecting search metrics.  
- Everything runs in the notebook, no separate `requirements.txt` is required.

---

## Link to GitHub Repository

[JB_semantic_search Repository](https://github.com/Nik002v/JB_semantic_search/tree/main)
