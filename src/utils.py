import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os
from tqdm import tqdm
from .download_and_process_datasets import CATCOLS
from .metrics import RowDifference
from scipy.optimize import linear_sum_assignment

def calc_diffusion_params(beta_1, beta_T, T):
    betas = np.linspace(beta_1, beta_T, T)
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas, axis=0)
    diffusion_config = {"betas": betas, "alphas": alphas, "alpha_bars": alpha_bars}
    return diffusion_config


class Preprocessor:
    def __init__(self, dataname, size=None, random_state=42):
        self.dataset = dataname
        self.OneHotEncoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # ignore unseen categories
        self.df = pd.read_csv(f"datasets/{dataname}/data.csv")
        self.df_train = pd.read_csv(f"datasets/{dataname}/train.csv")
        self.df_test = pd.read_csv(f"datasets/{dataname}/test.csv")
        
        self.cat_cols = CATCOLS[dataname]
        self.num_cols = [col for col in self.df.columns if col not in CATCOLS[dataname]]
        self.cat_col_indices = self.df.columns.get_indexer(self.cat_cols)
        self.num_col_indices = self.df.columns.get_indexer(self.num_cols)
        self.OneHotEncoder.fit(self.df[self.cat_cols])
        self.numeric_mean = np.mean(self.df_train[self.num_cols].values, axis=0)
        self.numeric_std = np.std(self.df_train[self.num_cols].values, axis=0)
        if size is not None:
            size_train = min(size, self.df_train.shape[0])
            size_test = min(size, self.df_test.shape[0])
            random_state = random_state
            self.df_train = self.df_train.sample(n=size_train, random_state=random_state).reset_index(drop=True)
            self.df_test = self.df_test.sample(n=size_test, random_state=random_state).reset_index(drop=True)

    def encodeDfToNp(self, df):
        cats = self.OneHotEncoder.transform(df[self.cat_cols])
        nums = df[self.num_cols].values
        return np.concatenate((nums, cats), axis=1)

    def standardize_np(self, arr):  # assume this array has format (nums, cats)
        # guard div-by-zero for constant cols
        temp = np.divide(arr[:, : len(self.num_cols)] - self.numeric_mean, self.numeric_std, out=np.zeros_like(arr[:, : len(self.num_cols)]), where=self.numeric_std!=0)
        return np.concatenate((temp, arr[:, len(self.num_cols) :]), axis=1)

    def destandardize_np(self, arr):
        temp = (arr[:, : len(self.num_cols)] * self.numeric_std) + self.numeric_mean
        return np.concatenate((temp, arr[:, len(self.num_cols) :]), axis=1)

    def decodeNpToDf(self, arr):
        cats_decoded = self.OneHotEncoder.inverse_transform(arr[:, len(self.num_cols) :])
        ordered = np.concatenate((arr[:, : len(self.num_cols)], cats_decoded), axis=1)
        reordered = ordered.copy()
        reordered[:, self.num_col_indices] = ordered[:, : len(self.num_cols)]
        reordered[:, self.cat_col_indices] = ordered[:, len(self.num_cols) :]
        df = pd.DataFrame(reordered, columns=self.df.columns)
        return df

class Matcher:
    
    def __init__(self, cat_cols, num_cols):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.numeric_std = 0
        self.differencer = RowDifference(cat_cols, num_cols, 0)

    def update_numeric_std(self, numeric_std):
        self.numeric_std = numeric_std
        self.differencer.update_numeric_std(numeric_std)

    def row_matching(self, target_batch, to_change):
        cost_matrix = self.cost_matrix(target_batch, to_change)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind, col_ind

    def greedy_matching(self, target_batch, to_change):
        """Greedy row assignment on the SAME cost matrix as row_matching (Hungarian).

        Repeatedly picks the global-minimum-cost (target, recon) pair, then masks its row
        and column. Returns (row_ind, col_ind) with row_ind = arange(N) and col_ind[i] the
        recon row assigned to target i -- same interface as row_matching. Mirrors the greedy
        `fast_align` heuristic used in ensembling, for the alignment-bias diagnostic (2.4).
        """
        cost = np.array(self.cost_matrix(target_batch, to_change), dtype=np.float64, copy=True)
        n_rows, n_cols = cost.shape
        col_ind = np.full(n_rows, -1, dtype=int)
        for _ in range(n_rows):
            idx = int(np.argmin(cost))
            i, j = divmod(idx, n_cols)
            col_ind[i] = j
            cost[i, :] = np.inf
            cost[:, j] = np.inf
        return np.arange(n_rows), col_ind

    def cost_matrix(self, target_batch, reconstructed_batch):
        return self.differencer.row_difference_matrix(target_batch, reconstructed_batch)
    
    def sim_matrix(self, target_batch, reconstructed_batch):
        return 1 - self.cost_matrix(target_batch, reconstructed_batch)


NUMERIC_KEYS = {
    "tableak_score" : "Accuracy",
    "reconstruction_time" : "Time",
    "dcr" : "DCR",
    "nndr" : "NNDR",
    "t_mae" : "t MAE",
    "mask_accuracy" : "Mask Acc",
    "noise_mse" : "Noise MSE",
}

UNITS = {
    "tableak_score" : "%",
    "reconstruction_time" : "seconds",
    "dcr" : "",
    "nndr" : "",
    "t_mae" : "",
    "mask_accuracy" : "",
    "noise_mse" : "",
}


class PerformanceManager:

    def __init__(self):
        self.table = {}
        self.reset()

    def reset(self):
        self.current_row = {}

    def log_metric(self, name, value):
        self.current_row[name] = value

    def commit_row(self, verbose=False):
        for key in self.table:
            if key not in self.current_row: self.current_row[key] = None
        for key in self.current_row:
            if key not in self.table:
                self.table[key] = []
            self.table[key].append(self.current_row[key])
        if verbose:
            tqdm.write(f"Logged metrics: {self.current_row}")
        self.reset()

    def save(self, filepath, append=False):
        df = self.to_pd()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if append:
            if os.path.exists(filepath):
                # Read only the header row to check for column mismatch
                existing_cols = list(pd.read_csv(filepath, nrows=0).columns)
                if set(existing_cols) == set(df.columns):
                    # Columns match — fast append (reorder to match existing header)
                    df[existing_cols].to_csv(filepath, mode='a', header=False, index=False)
                else:
                    # Column mismatch — fall back to read-concat-write
                    old_df = pd.read_csv(filepath)
                    combined_df = pd.concat([old_df, df], ignore_index=True)
                    combined_df.to_csv(filepath, index=False)
            else:
                df.to_csv(filepath, index=False)
            self.table = {}
        else:
            df.to_csv(filepath, index=False)

    def __repr__(self) -> str:
        # show avg of each column
        repr_str = "Performance Summary:\n"
        for key in self.table:
            # if column is numeric, show average
            if key in NUMERIC_KEYS:
                if None in self.table[key]:
                    repr_str += f"{NUMERIC_KEYS[key]}: N/A\n"
                else:
                    avg_value = np.mean(self.table[key])
                    if UNITS[key] == "%":
                        repr_str += f"{NUMERIC_KEYS[key]}: {avg_value*100:<.2f}{UNITS[key]}\n"
                    elif UNITS[key] == "seconds":
                        repr_str += f"{NUMERIC_KEYS[key]}: {avg_value:<.2f} {UNITS[key]}\n"
                    else:
                        repr_str += f"{NUMERIC_KEYS[key]}: {avg_value:<.2f}\n"


        return repr_str

    def to_pd(self):
        return pd.DataFrame(self.table)
