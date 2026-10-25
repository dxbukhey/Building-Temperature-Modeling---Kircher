# Building-Temperature-Modeling---Kircher
Making a prediction on indoor temperatures, comparing five modeling techniques

1. Kircher_Research_Full_Project.pptx — concise project presentation and findings
2. final_verify.py — consolidated analysis, model training, validation, and figure generation
3. upof-model-input-20260512-20260708.csv — original training/validation dataset
4. final_verified_results.csv — five-model results across forecast horizons
5. final_temperature_plot.png — measured vs. predicted temperatures
6. final_feature_importance.csv — XGBoost feature importance
7. lstm_seed_robustness.csv — LSTM random-seed checks

Note: final_verify.py assumes files live in /mnt/data/ and requires numpy, pandas, torch, scikit-learn, xgboost, and matplotlib. Edit P in the script to run in another folder.
Held-out test data is not included.
