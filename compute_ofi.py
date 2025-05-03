import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def preprocess_ofi_data(df):
    """Convert timestamp columns to pandas datetime."""
    df['ts_event'] = pd.to_datetime(df['ts_event'].str.rstrip('Z'))
    df['ts_recv'] = pd.to_datetime(df['ts_recv'].str.rstrip('Z'))
    return df

def compute_per_event_ofi(df, depth=10):
    """Compute order flow imbalance (OFI) at each level."""
    ofi_cols = []

    for m in range(depth):
        px_bid = f'bid_px_{m:02d}'
        px_ask = f'ask_px_{m:02d}'
        sz_bid = f'bid_sz_{m:02d}'
        sz_ask = f'ask_sz_{m:02d}'

        px_bid_diff = df[px_bid].diff()
        px_ask_diff = df[px_ask].diff()
        sz_bid_diff = df[sz_bid].diff()
        sz_ask_diff = df[sz_ask].diff()

        # OFI logic: positive for aggressive buy, negative for aggressive sell
        # Note: corrected a typo in Section 2.1 of the paper.
        # The negative sign should be applied to the previous row's size (shifted), not the current one.
        of_bid = np.where(px_bid_diff > 0, df[sz_bid],
                 np.where(px_bid_diff < 0, -df[sz_bid].shift(1), sz_bid_diff))
        of_ask = np.where(px_ask_diff > 0, -df[sz_ask].shift(1),
                 np.where(px_ask_diff < 0, df[sz_ask], sz_ask_diff))

        ofi = of_bid - of_ask
        col = f'ofi_{m:02d}'
        df[col] = ofi
        ofi_cols.append(col)

    return df, ofi_cols

def compute_cumulative_ofi(df, ofi_cols, depth=10, h='1s'):
    """
    Compute normalized cumulative OFI for all levels.
    """
    df = df.set_index('ts_event')

    # Rolling sum for each level
    for col in ofi_cols:
        df[f'{col}_cum'] = df[col].rolling(h).sum()

    # Rolling average book depth across all levels
    q_sum = None
    for m in range(depth):
        bid_sz = f'bid_sz_{m:02d}'
        ask_sz = f'ask_sz_{m:02d}'
        q_m = (df[bid_sz] + df[ask_sz]) / 2
        q_m_avg = q_m.rolling(h).mean()
        q_sum = q_m_avg if q_sum is None else q_sum + q_m_avg

    Q_M_h = q_sum / depth  # average across all levels

    # Scale OFI using depth
    ofi_cum_cols = []
    for col in ofi_cols:
        cum_col = f'{col}_cum'
        df[cum_col] = df[cum_col] / Q_M_h
        ofi_cum_cols.append(cum_col)

    df.reset_index(inplace=True)
    df.drop(columns=ofi_cols, inplace=True)

    return df, ofi_cum_cols

def compute_integrated_ofi(df, ofi_cum_cols):
    """
    Perform PCA on historical OFI values and compute integrated OFI score.
    """
    integrated_ofis = [np.nan, np.nan, np.nan]  # Padding due to rolling window and PCA lag

    for t in range(3, len(df)):
        X_hist = df.iloc[1:t][ofi_cum_cols].values
        x_curr = df.iloc[t][ofi_cum_cols].values.reshape(1, -1)

        pca = PCA(n_components=1)
        pca.fit(X_hist)
        w1 = pca.components_[0]
        w1_normalized = w1 / np.sum(np.abs(w1))  # L1 normalization

        integrated = np.dot(w1_normalized, x_curr.ravel())
        integrated_ofis.append(integrated)

    df = df.copy()
    df['ofi_integrated'] = integrated_ofis
    return df

def compute_all_ofi(df, depth=10, h='1s'):
    """
    Run full OFI feature engineering pipeline.
    """
    df = preprocess_ofi_data(df)
    df, ofi_cols = compute_per_event_ofi(df, depth=depth)
    df, ofi_cum_cols = compute_cumulative_ofi(df, ofi_cols, depth=depth, h=h)
    df = compute_integrated_ofi(df, ofi_cum_cols)
    return df, ofi_cum_cols