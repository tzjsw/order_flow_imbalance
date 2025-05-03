# Order Flow Imbalance (OFI) Feature Engineering

This repository contains code to compute Order Flow Imbalance (OFI) features from Limit Order Book (LOB) data, including cumulative and integrated OFI using Principal Component Analysis (PCA). It is based on *Cont, R., Cucuringu, M., & Zhang, C. (2023). Cross-impact of order flow imbalance in equity markets. Quantitative Finance, 23(10), 1373–1393. https://doi.org/10.1080/14697688.2023.2236159.*
## Files

- `compute_ofi.py`: Contains the core Python functions for preprocessing data, computing OFI at each level, cumulative OFI, and an integrated OFI using PCA.
- `ofi_pipeline.ipynb`: A Jupyter notebook demonstrating how to load data and apply the full OFI feature pipeline.

## Features

- **Per-event OFI**: Measures order book imbalance at each price level.
- **Cumulative OFI**: Normalized rolling sum of OFI over a specified time window.
- **Integrated OFI**: Dimensionality-reduced OFI using PCA to extract the principal direction of order flow pressure.

