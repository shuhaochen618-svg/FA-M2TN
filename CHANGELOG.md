# Changelog

## 2.0.0 - August 2026 revision

- Align the public implementation with the final August PG-M2TN protocol.
- Use charge-only profiles with dataset-stratified cell-level splits.
- Fix the main architecture at `hidden_dim=128`, two BiLSTM layers, and 630,149 parameters.
- Use fixed SOH/VDR/MAE loss coefficients of 0.50/0.50/0.10.
- Add exact Full, No-MAE, No-VDR, and SOH-only training variants.
- Add deterministic validation and test masking and held-out test evaluation.
- Remove legacy dynamic task weighting, alpha/IC routing, full-cycle compatibility, and baseline-model code.
- Rewrite the reproduction instructions and document the August reference results.
