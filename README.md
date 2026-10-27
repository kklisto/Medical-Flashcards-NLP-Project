# Medical NLP: retrieval and question answering

Academic team project exploring how to organise medical flashcards, retrieve relevant answers and compare retrieval with generative language models.

**Authors:** Juliette Treyer, Tanguy Rolland, Kalisto Willaey and Théo Moreau (Team Baguette, 2024). The notebook records the team's work; it does not attribute each component to an individual author.

## Start here

Open the [project notebook](Project_NLP_Team_Baguette.ipynb). For a focused technical review, start at **Question Answering using Sentence Transformers**, then **Approximated Nearest Neighbor search** and **Re-ranking**.

```text
Medical flashcards → text embeddings → HNSW candidate retrieval
                                      → cross-encoder reranking → retrieved answer
```

## Approaches explored

| Track | Implementation |
| --- | --- |
| Corpus analysis | TF-IDF, K-Means/MiniBatchKMeans, SVD, LDA and visualisation |
| Word embeddings | Word2Vec trained on the flashcard text |
| Sparse retrieval | BM25 with PyTerrier |
| Dense retrieval | `multi-qa-MiniLM-L6-cos-v1` embeddings and cosine similarity |
| Retrieve and rerank | HNSW candidate indexing and `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generative experiments | GPT-2 adaptation, a quantised LoRA experiment and Phi-3-mini inference |
| Evaluation | Sentence embedding similarity, BERTScore and qualitative answer inspection |

The dense-retrieval section holds out 200 flashcards for evaluation. Other notebook sections use their own evaluation arrangements; their scores should not be treated as a single controlled benchmark.

## Data and execution

The notebook expects `medical_meadow_wikidoc_medical_flashcards.json`, with `instruction`, `input` and `output` fields. Its exploration section reports 33,547 flashcards after cleaning. The dataset and fine-tuned model checkpoints are not included in this repository.

1. Obtain the original Medical Meadow / WikiDoc flashcard dataset and check its usage terms.
2. Open the notebook in Jupyter, Colab or a compatible notebook environment.
3. Replace the Google Drive, Kaggle and relative dataset paths with your own paths. Skip the Colab Drive mount outside Colab.
4. Run one experiment section at a time, including its prerequisite data-preparation cells. The notebook was assembled across environments and is not a single turnkey execution pipeline.
5. Install the dependencies required by that section. The retrieval track uses NumPy, pandas, scikit-learn, matplotlib, seaborn, sentence-transformers and hnswlib. Other tracks additionally use gensim, PyTerrier, transformers, PyTorch, evaluate, bert-score, PEFT, bitsandbytes and accelerate.

PyTerrier requires a compatible Java environment. Generative and quantised experiments have additional hardware and package compatibility requirements. Original dependency versions are not locked. Some cells load local checkpoints that must first be produced by the corresponding training section; flags control whether training or evaluation runs. Inspect notebook installation cells and model-loading options before execution.

## Findings and limitations

The project makes it possible to inspect the trade-offs between lexical retrieval, embedding search, reranking and generation. The notebook's qualitative evaluation also documents coherent but incorrect generated answers, which is a useful reminder that fluency and embedding similarity do not establish factual correctness.

Saved outputs are historical experiment records, not newly reproduced results. Data-cleaning logic and split construction vary between sections, and similarity metrics do not validate medical accuracy. This is an educational project, with no clinical validation or deployment claim.
