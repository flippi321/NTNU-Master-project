import json
import os
import numpy as np
import pandas as pd


class StratifiedSplitter:
    """
    Iterative multi-label stratified split (Sechidis et al. 2011).

    Accepts any normalized DataFrame (health data, FastSurfer data, or
    combined) indexed or containing a subject-ID column. Continuous features
    are binned into up to 4 quantile bins; binary features are kept as-is.
    All stratification signals are derived automatically from the data.

    Typical usage
    -------------
    splitter = StratifiedSplitter()
    train_ids, val_ids, test_ids = splitter.split(df)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self,
        df:             pd.DataFrame,
        id_col:         str             = "hunt_id",
        skip_cols:      set[str] | None = None,
        train_split:    float           = 0.70,
        val_split:      float           = 0.15,
        seed:           int             = 69,
        save_path:      str | None      = None,
        filter_by_mri:  bool            = True,
        mri_root:       str             = "data/",
    ) -> tuple[list, list, list]:
        """
        Partition subjects into train / val / test sets.

        Continuous columns (nunique > 2) are binned into up to 4 quantile
        bins, each represented as a binary indicator. Binary columns are kept
        as-is. The algorithm assigns subjects so that every signal is evenly
        represented across all three splits.

        Parameters
        ----------
        df : pd.DataFrame
            Normalized feature table. Must contain ``id_col`` as a column or
            as the index.
        id_col : str
            Name of the subject-ID column (default ``"hunt_id"``).
        skip_cols : set of str, optional
            Extra columns to exclude from stratification signals.
            Defaults to ``{"long_id", "mr_hunt_id"}``.
        train_split : float
            Fraction of subjects for training (default 0.70).
        val_split : float
            Fraction for validation (default 0.15). Test gets the remainder.
        seed : int
            Random seed for reproducibility (default 69).
        save_path : str, optional
            If given, write the split assignment to this JSON file.
        filter_by_mri : bool
            If True (default), drop any row whose ``id_col`` is not present
            as a sub-directory in *both* ``<mri_root>/HUNT3`` and
            ``<mri_root>/HUNT4``. Prevents splits from referencing subjects
            without MRI files on disk.
        mri_root : str
            Root containing the ``HUNT3/`` and ``HUNT4/`` MRI directories
            (default ``"data/"``). Only used when ``filter_by_mri`` is True.

        Returns
        -------
        train_ids, val_ids, test_ids : lists of subject-ID strings.
        """
        assert train_split + val_split < 1.0, "train_split + val_split must be < 1.0"

        if df.index.name == id_col:
            df = df.reset_index()

        if filter_by_mri:
            df = self._filter_by_mri(df, id_col, mri_root)

        skip = (skip_cols if skip_cols is not None else {"long_id", "mr_hunt_id"}) | {id_col}

        label_df     = self._build_label_matrix(df, id_col, skip)
        hunt_ids     = label_df.index.to_numpy()
        label_matrix = label_df.values.astype(float)
        n            = len(hunt_ids)

        rng   = np.random.default_rng(seed)
        order = rng.permutation(n)
        hunt_ids     = hunt_ids[order]
        label_matrix = label_matrix[order]

        proportions       = np.array([train_split, val_split, 1.0 - train_split - val_split])
        desired_per_label = np.outer(proportions, label_matrix.sum(axis=0))  # (3, L)
        split_budget      = (n * proportions).astype(float)

        assignments = np.full(n, -1, dtype=int)
        unassigned  = set(range(n))

        while unassigned:
            rem        = np.array(list(unassigned))
            label_sums = label_matrix[rem].sum(axis=0)
            present    = np.where(label_sums > 0)[0]

            if len(present) == 0:
                # No label evidence left — fill by budget
                for idx in rem:
                    k = int(np.argmax(split_budget))
                    assignments[idx] = k
                    split_budget[k] -= 1
                break

            # --- Algorithm 1, line 14: rarest label ---
            counts_present = label_sums[present]
            min_count      = counts_present.min()
            tied           = present[counts_present == min_count]
            rarest_j       = int(rng.choice(tied))

            # --- Algorithm 1, lines 15–33: assign carriers ---
            carriers = rem[label_matrix[rem, rarest_j] == 1]
            for chosen in carriers:
                chosen = int(chosen)

                label_desires    = desired_per_label[:, rarest_j]
                max_label_desire = label_desires.max()
                top_subsets      = np.where(label_desires == max_label_desire)[0]

                if len(top_subsets) == 1:
                    k = int(top_subsets[0])
                else:
                    budgets    = split_budget[top_subsets]
                    max_budget = budgets.max()
                    top2       = top_subsets[budgets == max_budget]
                    k          = int(rng.choice(top2))

                assignments[chosen] = k
                unassigned.remove(chosen)

                desired_per_label[k] -= label_matrix[chosen]
                desired_per_label     = np.maximum(desired_per_label, 0)
                split_budget[k]      -= 1

        train_ids = hunt_ids[assignments == 0].tolist()
        val_ids   = hunt_ids[assignments == 1].tolist()
        test_ids  = hunt_ids[assignments == 2].tolist()

        if save_path:
            record = {
                "seed":        seed,
                "train_split": train_split,
                "val_split":   val_split,
                "train":       train_ids,
                "val":         val_ids,
                "test":        test_ids,
            }
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(record, f, indent=2)
            print(f"[StratifiedSplitter] Split saved → {save_path}")

        return train_ids, val_ids, test_ids

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _filter_by_mri(self, df: pd.DataFrame, id_col: str, mri_root: str) -> pd.DataFrame:
        """Keep only rows whose id appears as a sub-directory in both HUNT3/ and HUNT4/."""
        hunt3_dir = os.path.join(mri_root, "HUNT3")
        hunt4_dir = os.path.join(mri_root, "HUNT4")
        for d in (hunt3_dir, hunt4_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(
                    f"[StratifiedSplitter] MRI directory {d!r} not found. "
                    "Pass filter_by_mri=False or set mri_root correctly."
                )
        available = set(os.listdir(hunt3_dir)) & set(os.listdir(hunt4_dir))

        n_before = len(df)
        id_str = df[id_col].astype(str).str.zfill(5)
        df = df[id_str.isin(available)].reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"[StratifiedSplitter] Dropped {n_dropped}/{n_before} subjects without MRI on disk → {len(df)} remain")
        return df

    def _build_label_matrix(self, df: pd.DataFrame, id_col: str, skip: set[str]) -> pd.DataFrame:
        df   = df.set_index(id_col)
        cols = []

        def _add(series, name):
            cols.append(series.rename(name))

        def _bin(series, prefix):
            bins = pd.qcut(series, q=4, labels=False, duplicates="drop")
            for b in range(int(bins.nunique())):
                indicator = (bins == b).fillna(False).astype(int)
                _add(indicator, f"{prefix}_bin_{b}")

        for col in df.columns:
            if col in skip:
                continue
            series   = df[col]
            n_unique = series.nunique()
            if n_unique < 2:
                continue
            elif n_unique == 2:
                _add(series.fillna(0).astype(int), col)
            else:
                _bin(series, col)

        return pd.concat(cols, axis=1)
