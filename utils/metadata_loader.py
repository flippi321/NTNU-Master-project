import json
import os
import numpy as np
import pandas as pd


class MetaDataLoader:
    def __init__(self, root_path="data/metadata", external_features=False):
        """
        Parameters
        ----------
        root_path : str
            Directory containing HUNT3.csv, HUNT4.xlsx, and (optionally)
            mock_metadata_normalized.csv.
        external_features : bool
            If True, load and merge mock_metadata_normalized.csv so that
            external feature columns are available via get() and split().
            Default False.
        """
        self.data_root = root_path
        self._use_external = external_features
        self._combined = None
        self._index = {}
        self._external_cols = []

    def _get_id_from_path(self, path: str) -> str:
        filename = os.path.basename(path)
        return filename.split('_')[0]

    def _load(self):
        if self._combined is not None:
            return

        hunt3 = pd.read_csv(os.path.join(self.data_root, "HUNT3.csv"))
        hunt4 = pd.read_excel(os.path.join(self.data_root, "HUNT4.xlsx"))

        hunt3 = hunt3.rename(columns={
            "MR_HUNT_ID": "hunt_3_long_id",
            "Age_at_time_of_MRI": "age_hunt3"
        })
        hunt4 = hunt4.rename(columns={
            "HUNT4 MRI Participant number": "hunt4_id",
            "Long HUNT3 numbers": "hunt_3_long_id",
            "Sex": "sex",
            "Age at HUNT4 MRI": "age_hunt4"
        })

        self._combined = hunt4.merge(hunt3[["hunt_3_long_id", "age_hunt3"]], on="hunt_3_long_id", how="inner")

        self._combined["hunt_id"] = self._combined["hunt_3_long_id"].apply(lambda x: str(int(x))[-5:])
        self._combined["sex"] = self._combined["sex"].map({"M": 0, "F": 1})
        self._combined["age_hunt3"] = (self._combined["age_hunt3"] - self._combined["age_hunt3"].min()) / (self._combined["age_hunt3"].max() - self._combined["age_hunt3"].min())
        self._combined["age_hunt4"] = (self._combined["age_hunt4"] - self._combined["age_hunt4"].min()) / (self._combined["age_hunt4"].max() - self._combined["age_hunt4"].min())

        # Optionally merge normalised external features
        if self._use_external:
            ext_path = os.path.join(self.data_root, "mock_metadata_normalized.csv")
            ext = pd.read_csv(ext_path)
            ext = ext.rename(columns={"MR_HUNT_ID": "hunt_3_long_id"})
            self._external_cols = [c for c in ext.columns if c != "hunt_3_long_id"]
            self._combined = self._combined.merge(ext, on="hunt_3_long_id", how="left")

        # Build O(1) lookup index
        self._index = self._combined.set_index("hunt_id").to_dict(orient="index")

    def _build_label_matrix(self) -> pd.DataFrame:
        """
        Build a binary indicator matrix used by the feature-based split algorithm.

        Each row is a participant (hunt_id). Each column is a binary
        stratification signal:
          - Binary columns (nunique == 2) are kept as-is.
          - Categorical columns (nunique > 2) are one-hot encoded into separate binary
          - Continuous columns (nunique > 2) are binned into up to 4 quantile
            bins, each represented as a separate binary indicator column.

        Returns
        -------
        pd.DataFrame of dtype int, indexed by hunt_id.
        """
        self._load()
        df = self._combined.set_index("hunt_id")
        cols = []

        def _add_column(series, name):
            cols.append(series.rename(name))

        def _bin_continuous(series, prefix):
            bins = pd.qcut(series, q=4, labels=False, duplicates="drop")
            for b in range(int(bins.nunique())):
                indicator = (bins == b).fillna(False).astype(int)
                _add_column(indicator, f"{prefix}_bin_{b}")

        # --- Always-present features ---
        sex = df["sex"].fillna(0).astype(int)
        _add_column(sex, "sex")
        _bin_continuous(df["age_hunt3"], "age_hunt3")
        _bin_continuous(df["age_hunt4"], "age_hunt4")

        # --- External features (when loaded) ---
        for col in self._external_cols:
            series = df[col]
            n_unique = series.nunique()
            if n_unique < 2:
                continue  # Constant column — no stratification signal
            elif n_unique == 2:
                _add_column(series.fillna(0).astype(int), col)
            else:
                _bin_continuous(series, col)

        return pd.concat(cols, axis=1)

    def split(self, train_split: float = 0.70, val_split: float = 0.15, seed: int = 69):
        """
        Stratified split of all participants into train / val / test sets.

        Continuous features (age and any external continuous cols) are binned
        into up to 4 quantile bins. Iterative multi-label stratification
        (Sechidis et al. 2011) then assigns participants to splits so that
        every binary/one-hot value and every bin is evenly represented.

        A split.json file is written to self.data_root recording the assignment
        for reproducibility.

        Parameters
        ----------
        train_split : float
            Fraction of participants for training (default 0.70).
        val_split : float
            Fraction for validation (default 0.15). Test gets the remainder.
        seed : int
            Random seed (default 42).

        Returns
        -------
        train_ids, val_ids, test_ids : lists of 5-digit hunt_id strings.
        """
        assert train_split + val_split < 1.0, "train_split + val_split must be < 1.0"

        self._load()
        label_df = self._build_label_matrix()

        hunt_ids = label_df.index.to_numpy()
        label_matrix = label_df.values.astype(float)
        n = len(hunt_ids)

        # Shuffle for reproducibility
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        hunt_ids = hunt_ids[order]
        label_matrix = label_matrix[order]

        proportions = np.array([train_split, val_split, 1.0 - train_split - val_split])
        n_splits = 3

        # Desired positive-example counts per label per subset
        desired = np.outer(proportions, label_matrix.sum(axis=0))  # (3, L)

        # Remaining capacity per subset (float for smooth tie-breaking)
        split_budget = (n * proportions).astype(float)

        assignments = np.full(n, -1, dtype=int)
        remaining = list(range(n))

        while remaining:
            rem = np.array(remaining)
            label_sums = label_matrix[rem].sum(axis=0)  # (L,)
            present = np.where(label_sums > 0)[0]

            if len(present) == 0:
                # All remaining participants have no labels — distribute by budget
                for idx in remaining:
                    k = int(np.argmax(split_budget))
                    assignments[idx] = k
                    split_budget[k] -= 1
                break

            # Rarest label: lowest total desired across all subsets
            desired_totals = desired[:, present].sum(axis=0)
            rarest_j = present[int(np.argmin(desired_totals))]

            # First participant (already shuffled) that carries this label
            candidates_with_label = rem[label_matrix[rem, rarest_j] == 1]
            chosen = int(candidates_with_label[0])

            # Assign to the subset that needs this label most
            k = int(np.argmax(desired[:, rarest_j]))
            assignments[chosen] = k

            # Update desired and budget
            desired[k] -= label_matrix[chosen]
            desired = np.maximum(desired, 0)
            split_budget[k] -= 1

            remaining.remove(chosen)

        train_ids = hunt_ids[assignments == 0].tolist()
        val_ids   = hunt_ids[assignments == 1].tolist()
        test_ids  = hunt_ids[assignments == 2].tolist()

        # Persist the split for reproducibility
        split_record = {
            "seed": seed,
            "train_split": train_split,
            "val_split": val_split,
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }
        split_path = os.path.join(self.data_root, "split.json")
        with open(split_path, "w") as f:
            json.dump(split_record, f, indent=2)

        return train_ids, val_ids, test_ids

    def get(self, hunt_path, long_id=False, sex=False, age_hunt3=False, age_hunt4=False, labeled=False):
        self._load()
        hunt_id = self._get_id_from_path(hunt_path)
        row = self._index.get(hunt_id)
        if row is None:
            print(f"No metadata found for hunt_id: {hunt_id}")
            return None
        result = {}
        if long_id:   result["hunt4_id"]  = row["hunt4_id"]
        if sex:       result["sex"]        = row["sex"]
        if age_hunt3: result["age_hunt3"]  = row["age_hunt3"]
        if age_hunt4: result["age_hunt4"]  = row["age_hunt4"]
        if self._use_external:
            for col in self._external_cols:
                result[col] = row.get(col)
        return result if labeled else list(result.values())

    def get_many(self, hunt_paths, long_id=False, sex=False, age_hunt3=False, age_hunt4=False):
        return [self.get(hpath, long_id=long_id, sex=sex, age_hunt3=age_hunt3, age_hunt4=age_hunt4) for hpath in hunt_paths]