# JB Semantic Search

This repository contains a **semantic search system** for code and documents, including **fine-tuning**, **vector storage**, and **evaluation metrics**.  
The project runs entirely in a **single Colab notebook**, which can be executed multiple times with different configurations for experimentation.

It can be run **locally** or on **Google Colab** with a free **T4 GPU** for faster execution.

> 💡 **Tip:** Use the **Table of Contents tab in Colab** for faster navigation between sections.

---

## 📚 Table of Contents (Notebook Sections)

1. **LIBS**  
   Lists all Python libraries used for data processing, embeddings, and vector search.

2. **Config**  
   Configuration settings: fine-tuning hyperparameters, vector DB parameters, embedding options (full function vs. names only).

3. **Weaviate DB**  
   Setup, initialization, and batch insertion methods for the embedded Weaviate vector database.

4. **Document Schemas**  
   Defines document types and their structure for storage and retrieval.

5. **Search Engine**  
   Implements semantic search logic, indexing, embedding, and query handling.

6. **METRICS**  
   Evaluation metrics for retrieval performance: **Recall**, **MRR**, **NDCG**, etc.

7. **Fine-tuning**  
   Scripts and logic for fine-tuning embeddings on code and document datasets.

8. **Data Loading**  
   Loading, preprocessing, and splitting datasets for training and evaluation.

9. **Demo**  
   Example queries for **papers** and **code**, demonstrating the search system in action.

10. **Plot Losses**  
    Visualization of training loss curves.

11. **MAIN**  
    Entry point for orchestrating workflow; defines the main function and other orchestration logic.

12. **RUN**  
    Executes the entire pipeline: prints all results, generates plots, and runs evaluations.  
    > Sections above RUN only define functions, classes, and configurations. RUN executes everything.

13. **DISCUSSION**  
    Textual discussion of experiments, findings, and design choices:  
    - **Model Choice**  
    - **Loss Function Choice**  
    - **Effect of Using Function Names Only**  
    - **🔍 Effect of HNSW Hyperparameters on Retrieval Metrics**  
    - **📊 Results: Effect of HNSW Database Hyperparameters**  
    - **🏁 Conclusion**

---

## ⚡ Setup & Running

1. Open [Google Colab](https://colab.research.google.com/).  
2. Upload `semantic_search_project.ipynb` or open it directly from GitHub.  
3. Enable GPU:  
   - Runtime → Change runtime type → Hardware accelerator → **GPU (T4)**  
4. Run all cells.  

> **Tip:** You can run the notebook **multiple times with different configurations** to test fine-tuning settings, embedding strategies, or HNSW parameters.

---

## 🔧 Notes

- **Embeddings:** Controlled by `Config.func_name_embedding` → full function vs function name only.  
- **DB Hyperparameters:** Controlled by `Config.vector_index_config` → HNSW parameters affect search metrics, not embedding quality.  
- **All code runs in a single notebook**, no separate `requirements.txt` is needed.  
- **RUN** section produces all final results (prints, plots, evaluation tables). Sections above define classes, functions, and configurations only.
- Use **Colab Table of Contents tab** for **faster navigation** between sections.
