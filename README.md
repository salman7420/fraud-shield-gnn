# fraud-shield-gnn
Fraud detection on IEEE-CIS dataset — progressive modeling from XGBoost baseline to Graph Neural Networks (GraphSAGE, HGT)

## Project Goal
Demonstrate how fraud detection performance improves across 4 model versions,
progressing from a simple tabular baseline to a graph-aware deep learning model
that detects organized fraud rings invisible to traditional ML.

## Models
| Version | Model | Type | Dataset | Key Technique |
|---|---|---|---|---|
| v1 | XGBoost | Supervised baseline | V1 — -999 null fill, label encoded | scale_pos_weight for imbalance |
| v2 | CatBoost | Supervised | V2 — raw categoricals, no encoding | Native categorical handling |
| v3 | XGBoost + FE | Supervised + feature engineering | V3 — velocity features, SMOTE | SHAP-driven feature selection |
| v4 | GraphSAGE + HGT | Graph Neural Network | V4 — PyG HeteroData | Card/device/email graph structure |
