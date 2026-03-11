# ⚡ EV Dominance DSS — From Combustion to Electric

## Setup & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Folder Structure
```
ev_dss_app/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Dependencies
├── best_clf.joblib               # Best classifier (Random Forest)
├── best_reg.joblib               # Best regressor (Gradient Boosting)
├── scaler.joblib                 # StandardScaler
├── feature_cols.joblib           # Feature column order
├── model_meta.json               # Model results & metadata
├── data/
│   └── ev_vs_petrol_dataset_v3.csv
└── assets/
    ├── eda_market_share_dist.png
    ├── eda_class_balance.png
    ├── eda_temporal.png
    ├── eda_regional.png
    ├── eda_correlation.png
    ├── eda_gdp_scatter.png
    ├── eval_clf_comparison.png
    ├── eval_reg_comparison.png
    ├── eval_confusion_matrix.png
    └── eval_feature_importance.png
```

## Deployment (Streamlit Cloud)
1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io
3. Connect your repo and set `app.py` as the main file
4. Deploy!
