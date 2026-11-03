# Evaluation Report

Run `python src/train.py` to generate `artifacts/model_comparison_cv.csv` and `artifacts/evaluation.json`.

The report must include:
- Majority-class baseline and 5-fold CV comparison for accuracy, precision, recall, F1, ROC-AUC and PR-AUC.
- RandomizedSearchCV search space and selected parameters.
- Final untouched stratified holdout classification report and confusion matrix.
- ROC-AUC, PR-AUC, Brier score and calibration buckets.
- Error analysis: inspect false negatives and false positives by contract and tenure segments before submission.

No metric values are pre-filled because they must come from an actual run of this version of the pipeline.
