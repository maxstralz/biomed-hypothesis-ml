# GNN smoke test through 2019

This is a **pipeline test only**. It uses the current graph at
`data/graph/surgery.edges.pkl`, whose embedded metadata identifies it as a
small, historical test graph ending in 2019. Do not use its metrics, model, or
predictions as results for the final surgical study.

The time split is intentionally non-overlapping:

| Purpose | Historical graph available through | Labels formed in |
| --- | ---: | ---: |
| Train and in-period validation | 2013 | 2014-2016 |
| OOD validation | 2016 | 2017-2019 |

The current graph contains 245,086 positive pairs in 2014-2016. The training
sample therefore uses 816,954 pairs, which gives approximately 30% positives
(`245,086 / 816,954`). The OOD data is deliberately stratified to 5% positives:
uniform random sampling produced no positives in this sparse graph and cannot
be evaluated with ROC-AUC. Its prevalence is an evaluation convenience, not an
estimate of real-world link prevalence.

## 1. Create output directories

```bash
mkdir -p data/model/surgery/gnn logs/surgery/gnn
```

## 2. Generate training data: predict 2014-2016

```bash
python -m materials_concepts.model.create_data \
  --graph_path data/graph/surgery.edges.pkl \
  --data_path data/model/surgery/gnn/smoke_train_2013.pkl \
  --year_start_train 2013 \
  --year_delta 3 \
  --edges_used_train 816954 \
  --edges_used_test 0 \
  --train_val_split 0.8 \
  --min_links 1
```

Expected: approximately 30% positive labels across the training and
in-period-validation split.

## 3. Generate OOD validation data: predict 2017-2019

```bash
python -m materials_concepts.model.create_data \
  --graph_path data/graph/surgery.edges.pkl \
  --data_path data/model/surgery/gnn/smoke_validation_2016.pkl \
  --year_start_test 2016 \
  --year_delta 3 \
  --edges_used_train 0 \
  --edges_used_test 100000 \
  --test_positive_ratio 0.05 \
  --min_links 1
```

Expected: 5,000 positive and 95,000 negative labels. The explicit positive
ratio makes the OOD ROC-AUC calculable; the old uniform sampler produced only
negative labels here.

## 4. Create historical node features

The training features use five snapshots that end in 2013. The OOD features use
five snapshots that end in 2016. Neither uses future edges for its period.

```bash
python -m materials_concepts.model.combi.pre_compute \
  --graph_path data/graph/surgery.edges.pkl \
  --output_path data/model/surgery/gnn/smoke_features_2013.pkl.gz \
  --binary True \
  --years "[2009, 2010, 2011, 2012, 2013]"
```

```bash
python -m materials_concepts.model.combi.pre_compute \
  --graph_path data/graph/surgery.edges.pkl \
  --output_path data/model/surgery/gnn/smoke_features_2016.pkl.gz \
  --binary True \
  --years "[2012, 2013, 2014, 2015, 2016]"
```

## 5. Train two GraphSAGE epochs on the M1 CPU

This retains the paper architecture but limits training to two epochs. PyG's
neighbour-sampling GraphSAGE path produced invalid MPS indices on this workload,
so the smoke test deliberately uses CPU. It should log `device: cpu`.

```bash
python -m materials_concepts.model.gnn.train_pyg train \
  --graph_path data/graph/surgery.edges.pkl \
  --data_path data/model/surgery/gnn/smoke_train_2013.pkl \
  --v_features_path data/model/surgery/gnn/smoke_features_2013.pkl.gz \
  --year_start_train 2013 \
  --train "batch_size=4096,num_epochs=2,lr=1e-5,weight_decay=0,log_interval=1,eval_batch_size=4096,ood_eval_interval=1,num_workers=0,amp=false,grad_clip_norm=1.0" \
  --model "hidden_dim=256,out_dim=128,dropout=0.1,decoder=mlp,decoder_hidden_dim=256,decoder_dropout=0.1" \
  --sampling "fanout1=20,fanout2=15" \
  --ood_data_path data/model/surgery/gnn/smoke_validation_2016.pkl \
  --ood_year_start 2016 \
  --ood_features_path data/model/surgery/gnn/smoke_features_2016.pkl.gz \
  --save_model_path data/model/surgery/gnn/smoke_graphsage_2013 \
  --log_file logs/surgery/gnn/smoke_graphsage_2013.log \
  --device_name cpu
```

Success means the run prints both an in-period `Epoch: ... AUC: ...` line and
an `OOD | Epoch: ... AUC: ...` line, then saves checkpoints under
`data/model/surgery/gnn/smoke_graphsage_2013/`.

## Do not carry these artifacts into the final experiment

Once LLM concept extraction is complete through 2025, rebuild the final graph
from `surgery.pubmed.gpt-5-nano.llm.csv`, regenerate all data and feature files,
and train again using the final non-overlapping 2017-2019, 2020-2022, and
2023-2025 windows.
