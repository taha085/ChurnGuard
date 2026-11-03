# Monitoring Notes

## Daily / weekly operational checks
- Log timestamp, model version, predicted probability, risk bucket, and request outcome. Avoid storing direct identifiers unless an approved retention policy requires them.
- Weekly compare feature distributions against the training reference: tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod, InternetService, total_services, tenure_bucket and charges_per_tenure_month.
- Check missing-value rates, unseen-category frequency, probability distribution, high-risk volume and API error/latency rates.
- Monthly, once labels mature, compare precision, recall, PR-AUC, calibration and false-negative rates by contract and tenure segments.

## Retraining triggers
Consider retraining when a sustained feature-distribution shift is observed across two review periods, or when matured-label recall/PR-AUC or calibration degrades beyond a pre-agreed business tolerance. Confirm data quality and label definitions before retraining. Re-run the full validation protocol and retain a new model version.
