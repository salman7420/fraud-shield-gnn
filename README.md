# fraud-shield-gnn
Graph Neural Network fraud detection on IEEE-CIS dataset — GraphSAGE, HGT, XGBoost, Isolation Forest

## Models
| Model | Type | Dataset |
|---|---|---|
| Isolation Forest | Unsupervised anomaly detection | V2 — tabular + median imputation |
| XGBoost v1 | Supervised baseline | V1 — tabular + -999 imputation |
| XGBoost v2 | Supervised + graph features | V3 — tabular + degree/centrality |
| GraphSAGE | Graph Neural Network | V4 — PyG HeteroData |
| HGT | Heterogeneous Graph Transformer | V4 — PyG HeteroData |
