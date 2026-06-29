# Training the paper-style best models for the surgery corpus

This file summarizes the commands for training the two trainable components behind the paper's best-performing model:

1. **Concept embeddings model**  
   Paper name: `Concept embeddings (MatSciBERT)`  
   Surgery version: `Concept embeddings (PubMedBERT/BiomedBERT)`

2. **GraphSAGE GNN model**  
   Paper name: `GNN @ Features`

The paper's best overall model is not a separate neural network. It is a **mixture of GNN + concept embeddings**, produced by blending the prediction scores from these two trained models.

For the surgery corpus, the recommended temporal split is:

- Training cutoff: `2016`
  - model sees graph/history up to and including 2016
  - positive training links are concept pairs that become connected during 2017-2019
- Validation/OOD cutoff: `2019`
  - model sees graph/history up to and including 2019
  - validation positives are concept pairs that become connected during 2020-2022
- Final held-out test cutoff, optional: `2022`
  - model sees graph/history up to and including 2022
  - final positives are concept pairs that become connected during 2023-2025

Do not tune hyperparameters on the final 2022-to-2025 split. Use it only once the setup is fixed.

## Expected input files

These commands assume the following files already exist:

```text
data/table/surgery.pubmed.gpt-5-nano.llm.csv
data/table/surgery.pubmed.gpt-5-nano.llm.with-concepts.csv
data/graph/surgery.edges.pkl
data/table/lookup/surgery.lookup.csv
data/embeddings/surgery_pubmedbert/
```

If you rebuild `surgery.edges.pkl`, also rebuild `surgery.lookup.csv` and regenerate/average embeddings against that exact lookup. Otherwise node IDs can silently stop matching.

## 0. Create output directories

```bash
mkdir -p \
  data/model/surgery/best_models \
  data/model/surgery/semantic \
  data/model/surgery/gnn \
  logs/surgery/semantic \
  logs/surgery/gnn \
  logs/surgery/mixture
```

What this does:

- Creates directories for pair-label data, trained models, predictions, metrics, and logs.
- `mkdir -p` is safe if the directories already exist.

Flags:

- `-p`: create parent directories as needed and do not fail if the directory already exists.

## 1. Build the graph and lookup if needed

Skip this if `data/graph/surgery.edges.pkl` and `data/table/lookup/surgery.lookup.csv` are already current.

```bash
python -m materials_concepts.graph.build \
  --input_path data/table/surgery.pubmed.gpt-5-nano.llm.with-concepts.csv \
  --output_path data/graph/surgery.edges.pkl \
  --output_lookup_path data/table/lookup/surgery.lookup.csv \
  --colname llm_concepts \
  --min_occurence 3 \
  --min_words 2 \
  --max_words 6 \
  --min_length 2 \
  --include_elements False
```

What this does:

- Reads the PubMed works table with extracted LLM concepts.
- Filters concepts into a stable graph vocabulary.
- Creates a temporal co-occurrence graph where concepts are nodes and co-occurrences in abstracts are timestamped edges.
- Writes the node lookup table that maps concept labels to numeric node IDs.

Flags:

- `--input_path`: CSV containing abstracts, publication dates, and `llm_concepts`. Prefer the filtered `with-concepts` file so rows with missing or empty concept lists are removed before graph construction.
- `--output_path`: output pickle file for the temporal graph edge list.
- `--output_lookup_path`: output CSV mapping graph node IDs to concept labels.
- `--colname`: concept column to use; for the surgery pipeline this is `llm_concepts`.
- `--min_occurence`: keep only concepts appearing at least this many times.
- `--min_words`: minimum concept-label word count. We use `2` to remove very broad one-word concepts.
- `--max_words`: maximum concept-label word count.
- `--min_length`: minimum character length.
- `--include_elements`: materials-science chemistry feature; keep `False` for biomedical concepts.

## 2. Generate raw PubMedBERT concept embeddings

Skip this if `data/embeddings/surgery_pubmedbert/` already contains the raw per-abstract PubMedBERT embedding chunks for the current graph lookup.

```bash
python -m materials_concepts.word_embeddings.generate \
  --concepts_path data/table/surgery.pubmed.gpt-5-nano.llm.with-concepts.csv \
  --lookup_path data/table/lookup/surgery.lookup.csv \
  --output_path data/embeddings/surgery_pubmedbert \
  --concept_column llm_concepts \
  --embedding_model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
  --log_to_stdout True \
  --step_size 500 \
  --resume True \
  --device_name auto
```

What this does:

- Reads each abstract and its retained graph concepts.
- Runs PubMedBERT/BiomedBERT over the abstract text.
- Extracts contextual token embeddings for concepts in that abstract.
- Saves compressed embedding chunks under `data/embeddings/surgery_pubmedbert/`.
- Uses resume logic so existing completed chunks are skipped.

Flags:

- `--concepts_path`: works CSV with abstracts, publication dates, and extracted concepts. Prefer the filtered `with-concepts` file.
- `--lookup_path`: graph lookup table. This is important: only concepts that survived graph construction are embedded.
- `--output_path`: directory where raw per-abstract embedding chunks are written.
- `--concept_column`: concept column in the works table.
- `--embedding_model`: Hugging Face encoder used for biomedical contextual embeddings. This PubMedBERT/BiomedBERT checkpoint replaces MatSciBERT for the surgery corpus.
- `--log_to_stdout`: print progress to the terminal.
- `--step_size`: number of abstracts saved per compressed chunk. If interrupted, only the current unfinished chunk is lost.
- `--resume`: when `True`, skip work IDs already present in existing embedding chunks.
- `--device_name`: `auto` chooses CUDA if available, then Apple MPS if available, then CPU. You can set `mps`, `cuda`, or `cpu` explicitly.

Important:

- If you rebuild `surgery.lookup.csv`, use a fresh `--output_path` or delete/regenerate the old raw embeddings. Resume is by paper ID, not by concept vocabulary, so an outdated lookup can silently miss new graph concepts.

## 3. Average PubMedBERT concept embeddings into node embeddings

You said you already ran these scripts. They are included here for completeness.

### 3.1 Average embeddings up to 2016

```bash
python -m materials_concepts.word_embeddings.average_embs \
  --concepts_path data/table/surgery.pubmed.gpt-5-nano.llm.with-concepts.csv \
  --lookup_path data/table/lookup/surgery.lookup.csv \
  --filter_path data/table/lookup/surgery.lookup.csv \
  --embeddings_dir data/embeddings/surgery_pubmedbert \
  --output_path data/model/surgery/semantic/pubmedbert_2016.pkl.gz \
  --concept_column llm_concepts \
  --store_concepts_ids True \
  --until_year 2016 \
  --only_average_contained False
```

What this does:

- Converts per-abstract PubMedBERT concept embeddings into one averaged embedding per graph node.
- Uses only papers published up to and including 2016.
- Saves node-ID-keyed embeddings for training the semantic model.

Flags:

- `--concepts_path`: works table with publication dates and extracted concepts. Prefer the filtered `with-concepts` file.
- `--lookup_path`: graph lookup table; this defines the valid node vocabulary.
- `--filter_path`: concepts to keep in the averaged output. Here it is the same as the lookup.
- `--embeddings_dir`: directory containing raw per-abstract PubMedBERT embedding chunks from `word_embeddings.generate`.
- `--output_path`: compressed output file containing averaged concept/node embeddings.
- `--concept_column`: concept column in the works table.
- `--store_concepts_ids`: when `True`, save embeddings keyed by graph node ID rather than concept string. This is required for model training.
- `--until_year`: inclusive temporal cutoff.
- `--only_average_contained`: if `False`, also use fallback abstract-average embeddings when the normalized concept label is not verbatim in the abstract.

### 3.2 Average embeddings up to 2019

```bash
python -m materials_concepts.word_embeddings.average_embs \
  --concepts_path data/table/surgery.pubmed.gpt-5-nano.llm.with-concepts.csv \
  --lookup_path data/table/lookup/surgery.lookup.csv \
  --filter_path data/table/lookup/surgery.lookup.csv \
  --embeddings_dir data/embeddings/surgery_pubmedbert \
  --output_path data/model/surgery/semantic/pubmedbert_2019.pkl.gz \
  --concept_column llm_concepts \
  --store_concepts_ids True \
  --until_year 2019 \
  --only_average_contained False
```

What this does:

- Same as the 2016 command, but uses literature up to and including 2019.
- This file is used for validation/OOD evaluation on links appearing during 2020-2022.

Flags:

- Same as the 2016 command.
- `--until_year 2019`: creates the semantic node embedding snapshot available at the 2019 cutoff.

### 3.3 Average embeddings up to 2022, optional final test

```bash
python -m materials_concepts.word_embeddings.average_embs \
  --concepts_path data/table/surgery.pubmed.gpt-5-nano.llm.with-concepts.csv \
  --lookup_path data/table/lookup/surgery.lookup.csv \
  --filter_path data/table/lookup/surgery.lookup.csv \
  --embeddings_dir data/embeddings/surgery_pubmedbert \
  --output_path data/model/surgery/semantic/pubmedbert_2022.pkl.gz \
  --concept_column llm_concepts \
  --store_concepts_ids True \
  --until_year 2022 \
  --only_average_contained False
```

What this does:

- Same as above, but uses literature up to and including 2022.
- Use this only for final held-out evaluation on links appearing during 2023-2025.

Flags:

- Same as the 2016 command.
- `--until_year 2022`: creates the semantic node embedding snapshot available at the 2022 cutoff.

## 4. Create train/validation link-prediction pairs

```bash
python -m materials_concepts.model.create_data \
  --graph_path data/graph/surgery.edges.pkl \
  --data_path data/model/surgery/best_models/train_2016_eval_2019.pkl \
  --year_start_train 2016 \
  --year_start_test 2019 \
  --year_delta 3 \
  --edges_used_train 5000000 \
  --edges_used_test 200000 \
  --train_val_split 0.8 \
  --min_links 1 \
  --test_positive_ratio 0.05
```

What this does:

- Creates supervised link-prediction examples.
- Training examples ask: among concept pairs not connected by 2016, which become connected by 2019?
- Validation/OOD examples ask: among concept pairs not connected by 2019, which become connected by 2022?
- The output pickle is used by both the semantic model and the GNN.

Flags:

- `--graph_path`: temporal graph created by `graph.build`.
- `--data_path`: output pickle containing `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, and `y_test`.
- `--year_start_train`: training cutoff. Pairs already connected by this year are excluded.
- `--year_start_test`: OOD/evaluation cutoff.
- `--year_delta`: future prediction horizon in years.
- `--edges_used_train`: number of training candidate pairs to sample. The paper-style value is large; reduce for smoke tests.
- `--edges_used_test`: number of OOD/evaluation candidate pairs to sample. The original repo used 2,000,000, but 200,000 is much friendlier on a laptop.
- `--train_val_split`: fraction of the training-period examples used for `X_train`; the rest becomes `X_val`.
- `--min_links`: minimum number of co-occurrences required for a future edge to count as positive.
- `--test_positive_ratio`: forces the evaluation set to contain both classes. `0.05` means 5% positives and 95% negatives. AUC is still the main metric; precision/recall depend on this artificial prevalence.

## 5. Train model 1: semantic PubMedBERT concept-embedding model

```bash
python -m materials_concepts.model.combi.train \
  --data_path data/model/surgery/best_models/train_2016_eval_2019.pkl \
  --emb_f_train_path "" \
  --emb_f_test_path "" \
  --emb_c_train_path data/model/surgery/semantic/pubmedbert_2016.pkl.gz \
  --emb_c_test_path data/model/surgery/semantic/pubmedbert_2019.pkl.gz \
  --lr 0.001 \
  --batch_size 1000 \
  --num_epochs 15000 \
  --pos_ratio 0.3 \
  --layers "[1536, 1024, 819, 10, 1]" \
  --step_size 200 \
  --gamma 0.9 \
  --dropout 0.1 \
  --sliding_window 5 \
  --log_interval 200 \
  --eval_batch_size 10000 \
  --log_file logs/surgery/semantic/pubmedbert_2016.log \
  --save_model data/model/surgery/semantic/pubmedbert_2016.pt
```

What this does:

- Trains the paper's `Concept embeddings` model.
- Uses no graph-topology features.
- Uses the concatenated embeddings of two concept nodes as the input vector.
- In the paper this was MatSciBERT; here it is PubMedBERT/BiomedBERT.

Flags:

- `--data_path`: pair-label dataset from step 3.
- `--emb_f_train_path`, `--emb_f_test_path`: topological feature paths. Empty strings disable them, making this a pure semantic model.
- `--emb_c_train_path`: concept/node embeddings at the training cutoff, here 2016.
- `--emb_c_test_path`: concept/node embeddings at the validation/OOD cutoff, here 2019.
- `--lr`: learning rate. The paper reports `1e-3` for the concept-embedding model.
- `--batch_size`: number of sampled pairs per gradient step.
- `--num_epochs`: in this script, one epoch is one sampled batch update, not a full pass over all pairs.
- `--pos_ratio`: positive fraction in each sampled training batch. The paper used 30%.
- `--layers`: MLP architecture. `1536 = 768 + 768`, because each pair concatenates two PubMedBERT vectors.
- `--step_size`: learning-rate scheduler step interval.
- `--gamma`: learning-rate multiplier applied at each scheduler step.
- `--dropout`: dropout probability. The paper used `0.1`.
- `--sliding_window`: early-stopping window over logged AUC/loss values.
- `--log_interval`: evaluate/log every N epochs. Keep this below `--num_epochs`; otherwise no AUC will print.
- `--eval_batch_size`: number of evaluation pairs processed per inference chunk.
- `--log_file`: where training logs are written.
- `--save_model`: final trained model path.

## 6. Save semantic-model predictions for blending

```bash
python -m materials_concepts.model.combi.eval \
  --data_path data/model/surgery/best_models/train_2016_eval_2019.pkl \
  --emb_f_test_path "" \
  --emb_c_test_path data/model/surgery/semantic/pubmedbert_2019.pkl.gz \
  --layers "[1536, 1024, 819, 10, 1]" \
  --dropout 0.1 \
  --model_path data/model/surgery/semantic/pubmedbert_2016.pt \
  --csv_path data/model/surgery/best_models/semantic_thresholds_2019.csv \
  --pred_path data/model/surgery/best_models/semantic_predictions_2019.pkl.gz \
  --metrics_path data/model/surgery/best_models/semantic_metrics_2019.pkl \
  --chunk_size 10000
```

What this does:

- Evaluates the trained semantic model on `X_test/y_test`, here the 2019-to-2022 OOD split.
- Saves prediction scores so they can be blended with GNN prediction scores.

Flags:

- `--data_path`: pair-label data; this command uses `X_test/y_test` from the file.
- `--emb_f_test_path`: empty because this is not the topology-feature model.
- `--emb_c_test_path`: semantic node embeddings for the evaluation cutoff.
- `--layers`: must match the architecture used during training.
- `--dropout`: must be accepted by the model constructor; dropout is inactive during evaluation.
- `--model_path`: trained semantic model.
- `--csv_path`: threshold-sweep metrics output.
- `--pred_path`: compressed prediction-score output used for blending.
- `--metrics_path`: summary metrics output.
- `--chunk_size`: number of candidate pairs evaluated at once.

## 7. Precompute topological node features for the GNN

### 7.1 Features up to 2016

```bash
python -m materials_concepts.model.combi.pre_compute \
  --graph_path data/graph/surgery.edges.pkl \
  --output_path data/model/surgery/gnn/features_2016.pkl.gz \
  --binary True \
  --years "[2012, 2013, 2014, 2015, 2016]"
```

What this does:

- Computes node-level topological feature vectors from graph snapshots up to 2016.
- These are used as initial node features for GraphSAGE training.

Flags:

- `--graph_path`: temporal graph file.
- `--output_path`: compressed feature file.
- `--binary`: if `True`, use binary adjacency rather than weighted adjacency.
- `--years`: graph snapshot years used for features. Five years produce 10 features per node because the code computes degree-like and two-hop/path-count-like features per year.

### 7.2 Features up to 2019

```bash
python -m materials_concepts.model.combi.pre_compute \
  --graph_path data/graph/surgery.edges.pkl \
  --output_path data/model/surgery/gnn/features_2019.pkl.gz \
  --binary True \
  --years "[2015, 2016, 2017, 2018, 2019]"
```

What this does:

- Computes the same topological features at the 2019 evaluation cutoff.
- Used for OOD validation on links appearing during 2020-2022.

Flags:

- Same as the 2016 command.
- `--years "[2015, 2016, 2017, 2018, 2019]"`: the five-year snapshot ending at the 2019 cutoff.

### 7.3 Features up to 2022, optional final test

```bash
python -m materials_concepts.model.combi.pre_compute \
  --graph_path data/graph/surgery.edges.pkl \
  --output_path data/model/surgery/gnn/features_2022.pkl.gz \
  --binary True \
  --years "[2018, 2019, 2020, 2021, 2022]"
```

What this does:

- Computes topological features at the 2022 final-test cutoff.
- Use only after hyperparameters are fixed.

Flags:

- Same as the 2016 command.
- `--years "[2018, 2019, 2020, 2021, 2022]"`: the five-year snapshot ending at the 2022 cutoff.

## 8. Train model 2: GraphSAGE GNN

On an M1/M2 Mac, use CPU for this command. The PyTorch Geometric neighbor-sampling path has produced invalid indices on MPS in this project.

```bash
python -m materials_concepts.model.gnn.train_pyg train \
  --graph_path data/graph/surgery.edges.pkl \
  --data_path data/model/surgery/best_models/train_2016_eval_2019.pkl \
  --v_features_path data/model/surgery/gnn/features_2016.pkl.gz \
  --year_start_train 2016 \
  --train "batch_size=4096,num_epochs=30,lr=1e-5,weight_decay=0,log_interval=1,eval_batch_size=16384,ood_eval_interval=1,num_workers=0,amp=false,grad_clip_norm=1.0" \
  --model "hidden_dim=256,out_dim=128,dropout=0.1,decoder=mlp,decoder_hidden_dim=256,decoder_dropout=0.1" \
  --sampling "fanout1=20,fanout2=15" \
  --ood_data_path data/model/surgery/best_models/train_2016_eval_2019.pkl \
  --ood_year_start 2019 \
  --ood_features_path data/model/surgery/gnn/features_2019.pkl.gz \
  --save_model_path data/model/surgery/gnn/graphsage_2016 \
  --log_file logs/surgery/gnn/graphsage_2016.log \
  --device_name cpu
```

What this does:

- Trains the paper-style two-layer GraphSAGE link-prediction model.
- Uses topological node features as input.
- Evaluates on both the in-period validation split and the 2019-to-2022 OOD split.
- Saves checkpoints and a final model under `data/model/surgery/gnn/graphsage_2016/`.

Flags:

- `--graph_path`: temporal graph file.
- `--data_path`: training pair-label data. Uses `X_train/y_train` and `X_val/y_val`.
- `--v_features_path`: topological node features available at the 2016 cutoff.
- `--year_start_train`: graph cutoff used for training message passing.
- `--train`: comma-separated training configuration:
  - `batch_size`: edge-label batch size.
  - `num_epochs`: number of training epochs. The paper reports architecture/hyperparameters, but not a strict epoch count; use validation/OOD AUC to decide.
  - `lr`: learning rate. The paper reports `1e-5`.
  - `weight_decay`: L2 regularization.
  - `log_interval`: evaluate every N epochs.
  - `eval_batch_size`: evaluation batch size.
  - `ood_eval_interval`: evaluate OOD every N epochs.
  - `num_workers`: PyTorch Geometric data-loader workers. Use `0` on Mac; use `8` or similar on a CUDA server.
  - `amp`: mixed precision. Use `false` on CPU; use `true` on CUDA.
  - `grad_clip_norm`: gradient clipping threshold.
- `--model`: comma-separated model configuration:
  - `hidden_dim=256`: GraphSAGE hidden size from the paper.
  - `out_dim=128`: final node embedding size from the paper.
  - `dropout=0.1`: dropout from the paper.
  - `decoder=mlp`: use an MLP decoder for link prediction.
  - `decoder_hidden_dim=256`: decoder hidden size from the paper.
  - `decoder_dropout=0.1`: decoder dropout.
- `--sampling`: GraphSAGE neighbor sampling:
  - `fanout1=20`: first-hop sampled neighbors.
  - `fanout2=15`: second-hop sampled neighbors.
- `--ood_data_path`: file containing `X_test/y_test` for OOD evaluation.
- `--ood_year_start`: graph cutoff for OOD message passing.
- `--ood_features_path`: topological features available at the OOD cutoff.
- `--save_model_path`: output directory for checkpoints, final model, and eval predictions.
- `--log_file`: training log path.
- `--device_name`: use `cpu` on Mac, `cuda` on an NVIDIA machine.

For an NVIDIA GPU machine, change the training config and device to:

```text
num_workers=8,amp=true
--device_name cuda
```

## 9. Save GNN predictions for blending

```bash
python -m materials_concepts.model.gnn.train_pyg eval \
  --graph_path data/graph/surgery.edges.pkl \
  --data_path data/model/surgery/best_models/train_2016_eval_2019.pkl \
  --v_features_path data/model/surgery/gnn/features_2019.pkl.gz \
  --model_path data/model/surgery/gnn/graphsage_2016/model.pt \
  --pred_path data/model/surgery/best_models/gnn_predictions_2019.pkl.gz \
  --split test \
  --year_start 2019 \
  --sampling "fanout1=20,fanout2=15" \
  --eval_batch_size 16384 \
  --num_workers 0 \
  --log_file logs/surgery/gnn/graphsage_2016_eval_2019.log \
  --device_name cpu
```

What this does:

- Loads the trained GraphSAGE model.
- Evaluates it on the same `X_test/y_test` split used by the semantic model.
- Saves prediction scores for blending.

Flags:

- `--graph_path`: temporal graph file.
- `--data_path`: pair-label data. With `--split test`, this command uses `X_test/y_test`.
- `--v_features_path`: topological features available at the evaluation cutoff, here 2019.
- `--model_path`: trained GNN checkpoint.
- `--pred_path`: compressed prediction-score output.
- `--split`: which split to evaluate. Use `test` for OOD validation.
- `--year_start`: graph cutoff used for message passing during evaluation.
- `--sampling`: must match the sampling shape used during training.
- `--eval_batch_size`: number of candidate edges evaluated per batch.
- `--num_workers`: data-loader workers. Use `0` on Mac; use more on CUDA/Linux.
- `--log_file`: evaluation log path.
- `--device_name`: use `cpu` on Mac, `cuda` on an NVIDIA machine.

## 10. Blend the two models: paper's best overall setup

```bash
python -m materials_concepts.model.mixture.blend \
  --data_path data/model/surgery/best_models/train_2016_eval_2019.pkl \
  --predictions_path_1 data/model/surgery/best_models/gnn_predictions_2019.pkl.gz \
  --predictions_path_2 data/model/surgery/best_models/semantic_predictions_2019.pkl.gz \
  --save_path data/model/surgery/best_models/mixture_gnn_semantic_predictions_2019.pkl.gz \
  --metrics_path data/model/surgery/best_models/mixture_gnn_semantic_metrics_2019.pkl \
  --details_path logs/surgery/mixture/gnn_semantic_blend_2019.txt
```

What this does:

- Blends GNN and semantic prediction scores.
- Tries weights from `0.0/1.0` to `1.0/0.0` in steps of `0.1`.
- Reports the best AUC and saves the blended predictions.
- This corresponds to the paper's best overall model family: `Mixture of GNN and Embeddings`.

Flags:

- `--data_path`: pair-label file containing `y_test`, used as ground truth for evaluating blend weights.
- `--predictions_path_1`: first prediction file, here GNN scores.
- `--predictions_path_2`: second prediction file, here semantic-model scores.
- `--save_path`: output blended prediction scores.
- `--metrics_path`: summary metrics for the best blend.
- `--details_path`: text file with AUC for every tested blend weight.

## Notes on interpreting metrics

- AUC is the most reliable metric for the sampled validation sets.
- Precision, recall, and F1 depend heavily on `--test_positive_ratio`.
- With `--test_positive_ratio 0.05`, the evaluation set has 5% positives. Natural graph prevalence is usually much lower.
- A high AUC means the model ranks future links well; it does not mean that threshold `0.5` is the best threshold for hypothesis recommendation.

## Quick smoke-test changes

For a fast check, reduce:

```text
--edges_used_train 200000
--edges_used_test 50000
```

For semantic training, reduce:

```text
--num_epochs 1000
--log_interval 50
```

For GNN training, reduce:

```text
num_epochs=2
```

Do not compare smoke-test metrics to paper-style metrics.
