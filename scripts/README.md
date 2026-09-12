### Capstone Project Script.
Four major scripts were used in the final submission to the **Agricultural Extension RAG: Smart Retrieval for Farmers Project Competition** and they are:
1. The finetuning used script on a BGE Reranker Large Model [(see here)](./fine-tuning-bge-reranker-model.ipynb)
2. The finetuning used script on a BAAI BGE Base model. [(see here)](./fine-tuning-bge-base-dense-model.ipynb)
3. A retrieval script using only the finetuned BGE reranker. [(see here)](./running-inference-on-fine-tune-of-bge-reranker.ipynb)
4. A retrieval script using hybrid retrieval where the retrieval output from the finetuned dense model are combined with the rankings of the BM25 algorithm to retrieve the top 50 outputs and then a reranker retrives the top ranking document from the 50 documented ranked my BM25 + finetuned dense embeddings model. [(see here)](./running-on-fine-tune-of-bge-rera-bge-dense-bm25.ipynb)

These scripts are highly dependent on the Kaggle Environment and must be ran there to access and import the finetuned models and the competition datasets.
To access the competition datasets you would need to be invitied to the competition by the organizers. 
>Contact **cohort-10@tri-ai.org** to get invited to the competition to enable you import the datasets to the notebooks.

2. The finetuned models are needed to perform retrieval using [the pure reranker](./running-inference-on-fine-tune-of-bge-reranker.ipynb) and [the hybrid retrieval](./running-on-fine-tune-of-bge-rera-bge-dense-bm25.ipynb) methods used by our team for this competition.

To import them you search for **BAAI-bge-base-finetune-agric-doc(tri-ai)** and **Bge reranker-large(tri-ai-agri)** when importing to the notebook.
You can also bookmark the models or add to a collection before hand to make the search process easier.
The models can be access used the links below:
1. [The Finetuned BGE Base Dense Embeddings model ](https://www.kaggle.com/models/israelolawuyi/baai-bge-base-finetune-agric-doctri-ai)
2. [The Finetuned BGE Large reranker model](https://www.kaggle.com/models/israelolawuyi/bge-reranker-largetri-ai-agri)
### Note

> To run the fine-tuning and pure reranking notebooks efficiently, it is recommended to use Kaggle GPUs such as the NVIDIA Tesla P100 or NVIDIA Tesla T4. This will improve fine-tuning and retrieval speeds.
