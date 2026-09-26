import numpy as np
SIGMA_CONSTANT = 0.319  # numerical tolerance factor (x feature std) from the TabLeak paper


class TableakSimilarity:

    def __init__(self, processor):
        self.cat_cols = processor.cat_cols
        self.num_cols = processor.num_cols
        self.cat_cols_indices = processor.cat_col_indices
        self.num_cols_indices = processor.num_col_indices
        self.numeric_mean = processor.numeric_mean
        self.numeric_std = processor.numeric_std
        self.tolerance = SIGMA_CONSTANT * np.array(self.numeric_std, dtype=np.float64)
        self.total_cols = len(self.cat_cols) + len(self.num_cols)
        # Per-categorical-column cardinality, aligned with self.cat_cols (the one-hot encoder
        # is fit on df[cat_cols], so categories_ preserves that order). Used to bin categorical
        # reconstruction accuracy by cardinality.
        ohe = getattr(processor, "OneHotEncoder", None)
        if ohe is not None and hasattr(ohe, "categories_"):
            self.cat_cardinalities = [len(c) for c in ohe.categories_]
        else:
            self.cat_cardinalities = [0] * len(self.cat_cols)

    # Cardinality bin edges: low <=4, medium 5-9, high >=10.
    def _cardinality_bin(self, card):
        if card <= 4:
            return "low"
        if card <= 9:
            return "medium"
        return "high"

    def tableak_similarity(self, batch_recon, batch_orig):
        # Vectorized: compute full N x M similarity matrix via broadcasting
        num_recon = batch_recon[self.num_cols].values.astype(np.float64)  # (N, n_num)
        num_orig = batch_orig[self.num_cols].values.astype(np.float64)   # (M, n_num)
        cat_recon = batch_recon[self.cat_cols].values  # (N, n_cat)
        cat_orig = batch_orig[self.cat_cols].values    # (M, n_cat)

        # Continuous: within tolerance
        within_tol = (np.abs(num_recon[:, None, :] - num_orig[None, :, :]) <= self.tolerance).sum(axis=2)  # (N, M)
        # Categorical: exact match
        cat_match = (cat_recon[:, None, :] == cat_orig[None, :, :]).sum(axis=2)  # (N, M)

        sim_matrix = (cat_match + within_tol) / self.total_cols  # (N, M)
        cat_score_matrix = cat_match / len(self.cat_cols)
        cont_score_matrix = within_tol / len(self.num_cols)

        # Greedy matching: for each recon row, pick best orig row
        N = sim_matrix.shape[0]
        result = 0.0
        result_cat = 0.0
        result_cont = 0.0
        for i in range(N):
            j = np.argmax(sim_matrix[i])
            result += sim_matrix[i, j]
            result_cat += cat_score_matrix[i, j]
            result_cont += cont_score_matrix[i, j]

        return result / N, result_cat / N, result_cont / N

    def tableak_similarity_linear(self, batch_recon, batch_orig):
        # Vectorized paired comparison (rows already aligned)
        num_recon = batch_recon[self.num_cols].values.astype(np.float64)
        num_orig = batch_orig[self.num_cols].values.astype(np.float64)
        cat_recon = batch_recon[self.cat_cols].values
        cat_orig = batch_orig[self.cat_cols].values

        within_tol = (np.abs(num_recon - num_orig) <= self.tolerance).sum(axis=1)  # (N,)
        cat_match = (cat_recon == cat_orig).sum(axis=1)  # (N,)

        scores = (cat_match + within_tol) / self.total_cols
        scores_cat = cat_match / len(self.cat_cols)
        scores_cont = within_tol / len(self.num_cols)

        N = len(batch_recon)
        return scores.sum() / N, scores_cat.sum() / N, scores_cont.sum() / N

    def detailed_similarity(self, batch_recon, batch_orig):
        """Aligned paired breakdown (rows already aligned: recon row i <-> orig row i).

        Returns a dict with:
          - per-cardinality categorical accuracy: cat_high/medium/low_score (None if the bin
            is empty for this dataset),
          - whole_row_acc: fraction of rows where EVERY cell is correct (cat exact + cont
            within tolerance) -- a strict record-level reconstruction metric,
          - near_row_80 / near_row_90: fraction of rows with at least 80% / 90% of cells correct,
          - per_column: {column_name -> per-row match rate} for every column.
        """
        num_recon = batch_recon[self.num_cols].values.astype(np.float64)
        num_orig = batch_orig[self.num_cols].values.astype(np.float64)
        cat_recon = batch_recon[self.cat_cols].values
        cat_orig = batch_orig[self.cat_cols].values

        cont_match = np.abs(num_recon - num_orig) <= self.tolerance  # (N, n_num) bool
        cat_match = cat_recon == cat_orig                            # (N, n_cat) bool

        # per-column match rate (averaged over rows)
        per_column = {}
        for k, col in enumerate(self.num_cols):
            per_column[col] = float(cont_match[:, k].mean()) if cont_match.shape[0] else 0.0
        for k, col in enumerate(self.cat_cols):
            per_column[col] = float(cat_match[:, k].mean()) if cat_match.shape[0] else 0.0

        # per-cardinality categorical bins
        bin_idxs = {"low": [], "medium": [], "high": []}
        for k, card in enumerate(self.cat_cardinalities):
            bin_idxs[self._cardinality_bin(card)].append(k)
        binned = {}
        for name, idxs in bin_idxs.items():
            binned[name] = float(cat_match[:, idxs].mean()) if idxs else None

        # whole-row strict exact-match over all columns
        if cat_match.shape[1] and cont_match.shape[1]:
            row_all = np.concatenate([cat_match, cont_match], axis=1)
        elif cat_match.shape[1]:
            row_all = cat_match
        else:
            row_all = cont_match
        whole_row_acc = float(row_all.all(axis=1).mean()) if row_all.shape[0] else 0.0
        # bridging metrics between whole-row (all cells) and mean per-cell accuracy:
        # fraction of records with at least 80% / 90% of cells correct.
        if row_all.shape[0]:
            frac_correct = row_all.mean(axis=1)  # per-record fraction of cells correct
            near_row_80 = float((frac_correct >= 0.8).mean())
            near_row_90 = float((frac_correct >= 0.9).mean())
        else:
            near_row_80 = near_row_90 = 0.0

        return {
            "cat_high_score": binned["high"],
            "cat_medium_score": binned["medium"],
            "cat_low_score": binned["low"],
            "whole_row_acc": whole_row_acc,
            "near_row_80": near_row_80,
            "near_row_90": near_row_90,
            "per_column": per_column,
        }

    def reidentification_rate(self, batch_recon, batch_orig):
        """Nearest-neighbour re-identification rate.

        For each reconstructed row, find its most-similar original row over the FULL batch;
        count a hit when that nearest neighbour is the row's own aligned match (index i).
        Rows must be aligned so orig index i is recon row i's true counterpart. Measures
        whether reconstructions are close enough to single out the correct private record.
        """
        num_recon = batch_recon[self.num_cols].values.astype(np.float64)
        num_orig = batch_orig[self.num_cols].values.astype(np.float64)
        cat_recon = batch_recon[self.cat_cols].values
        cat_orig = batch_orig[self.cat_cols].values

        within_tol = (np.abs(num_recon[:, None, :] - num_orig[None, :, :]) <= self.tolerance).sum(axis=2)
        cat_m = (cat_recon[:, None, :] == cat_orig[None, :, :]).sum(axis=2)
        sim_matrix = (cat_m + within_tol) / self.total_cols  # (N, M)
        N = sim_matrix.shape[0]
        if N == 0:
            return 0.0
        nn = np.argmax(sim_matrix, axis=1)
        return float((nn == np.arange(N)).mean())

    def tableak_similarity_row(self, row_recon, row_orig):
        score_cat = 0
        score_cont = 0

        for cat_col in self.cat_cols:
            if row_recon[cat_col] == row_orig[cat_col]:
                score_cat += 1

        for i, num_col in enumerate(self.num_cols):
            tolerance = self.tolerance[i]
            lbound = row_orig[num_col] - tolerance
            ubound = row_orig[num_col] + tolerance
            if lbound <= row_recon[num_col] <= ubound:
                score_cont += 1

        return (score_cat + score_cont) / self.total_cols, score_cat / len(self.cat_cols), score_cont / len(self.num_cols)

class RowDifference:

    def __init__(self, cat_cols, num_cols, numeric_std):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.numeric_std = numeric_std
        self.tolerance = SIGMA_CONSTANT * np.array(numeric_std, dtype=np.float64) if np.any(numeric_std != 0) else np.zeros(len(num_cols))

    def update_numeric_std(self, numeric_std):
        self.numeric_std = numeric_std
        self.tolerance = SIGMA_CONSTANT * np.array(numeric_std, dtype=np.float64)

    def row_difference_matrix(self, df_orig, df_recon):
        """Vectorized cost matrix: returns (N, M) array of row differences."""
        num_orig = df_orig[self.num_cols].values.astype(np.float64)   # (N, n_num)
        num_recon = df_recon[self.num_cols].values.astype(np.float64) # (M, n_num)
        cat_orig = df_orig[self.cat_cols].values   # (N, n_cat)
        cat_recon = df_recon[self.cat_cols].values  # (M, n_cat)

        total_cols = len(self.cat_cols) + len(self.num_cols)

        # Continuous mismatch: outside tolerance
        cont_mismatch = (np.abs(num_orig[:, None, :] - num_recon[None, :, :]) > self.tolerance).sum(axis=2)  # (N, M)
        # Categorical mismatch
        cat_mismatch = (cat_orig[:, None, :] != cat_recon[None, :, :]).sum(axis=2)  # (N, M)

        return (cat_mismatch + cont_mismatch) / total_cols

    def row_difference(self, row_orig, row_recon):
        cat_loss = 0
        cont_loss = 0
        for cat_col in self.cat_cols:
            if row_recon[cat_col] != row_orig[cat_col]:
                cat_loss += 1

        for i, num_col in enumerate(self.num_cols):
            tolerance = self.tolerance[i]
            lbound = row_orig[num_col] - tolerance
            ubound = row_orig[num_col] + tolerance
            if lbound > row_recon[num_col] or ubound < row_recon[num_col]:
                cont_loss += 1

        return (cat_loss + cont_loss) / (len(self.cat_cols) + len(self.num_cols))
