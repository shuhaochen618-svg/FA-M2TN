# Changelog

## 2.1.0 - FA-M2TN release

- Rename the public implementation to FA-M2TN: Fragmentation-Aware Masked Multi-Task Network.
- Align the public implementation with the reported experimental protocol.
- Use charge-only profiles with dataset-stratified cell-level splits.
- Fix the main architecture at `hidden_dim=128`, two BiLSTM layers, and 630,149 parameters.
- Use fixed SOH/VDR/MAE loss coefficients of 0.50/0.50/0.10.
- Add exact Full, No-MAE, No-VDR, and SOH-only training variants.
- Add deterministic validation and test masking and held-out test evaluation.
- Remove legacy dynamic task weighting, alpha/IC routing, full-cycle compatibility, and baseline-model code.
- Rewrite the reproduction instructions and document the reference results.
