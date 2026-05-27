# biomed-hypothesis-ml
LMM and ML in biomedical research hypotheses generation

## Installation

1. Install Miniconda or Anaconda if needed:
   - https://docs.conda.io/en/latest/miniconda.html

2. Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
```

3. Activate the environment:

```bash
conda activate biomed-hypothesis-ml
```

4. Verify the environment:

```bash
python -m pip show pandas numpy scikit-learn networkx requests openai tqdm
```

## Usage

- `python 01_abstract_mining.py` - download PubMed abstracts per module
- `python 02_extract_2_concepts.py` - extract two concepts per abstract using OpenAI
- `python 03_concept_graph.py` - build concept graph and edge summaries
- `python 04_link_prediction_data_set.py` - create link prediction dataset
- `python 06_concept_embedding.py` - generate concept embeddings
- `python 07_add_embedding_features_08.py` - add embedding features to the dataset
- `python 05_train_prediction_model.py` - train predictive models
- `python 08_predict_future_links.py` - predict future links
- `python 09_first_filter.py` - first hypothesis filtering
- `python 10_second_filter.py` - second hypothesis filtering
- `python 11_novelty_ranking.py` - rank final hypotheses by novelty
