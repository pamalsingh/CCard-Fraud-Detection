# Key Risks and Mitigation

| Risk | Suggested Mitigation |
|------|----------------------|
| Accuracy is misleading because the dataset is highly imbalanced | Validate assumptions early; use appropriate metrics (F1, PR-AUC, recall), resampling (SMOTE) or class-weighting; document limitations and include error analysis focusing on false negatives and false positives. |
| Anonymized PCA features limit business interpretability | Document feature transformation and limitations; supplement model outputs with error examples and aggregated feature-importance proxies (e.g., permutation importance) and work with domain experts to map insights back to business signals where possible. |
| Poor threshold selection can create too many false positives | Evaluate and tune decision threshold based on business cost/benefit (precision vs recall trade-off); present multiple operating points to stakeholders; include human-in-the-loop review for flagged transactions. |
| Responsible use concern | Design the system as an analyst-assist tool, not an automated adjudicator; require human review for adverse or high-impact actions; log decisions and provide explanations/samples for flagged cases. |

## Notes & Next Steps

- Include a short section in the model evaluation report that shows how metric selection and thresholding affect business outcomes (e.g., estimated workload for fraud analysts vs expected fraud captured).
- Keep these risks visible in the project README and presentation so stakeholders are aware of limitations and governance requirements.
