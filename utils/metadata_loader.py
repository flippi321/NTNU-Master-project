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
            Random seed (default 69).

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

        # Desired positive-example counts per label per subset (c^i_j in the paper)
        desired_per_label = np.outer(proportions, label_matrix.sum(axis=0))  # (3, L)

        # Desired total examples per subset (c_j in the paper)
        split_budget = (n * proportions).astype(float)

        assignments = np.full(n, -1, dtype=int)
        unassigned = set(range(n))

        while unassigned:
            rem = np.array(list(unassigned))

            # Count remaining examples per label among unassigned participants
            label_sums = label_matrix[rem].sum(axis=0)  # (L,)
            present = np.where(label_sums > 0)[0]

            if len(present) == 0:
                # No label evidence left — distribute remaining by subset budget
                for idx in rem:
                    k = int(np.argmax(split_budget))
                    assignments[idx] = k
                    split_budget[k] -= 1
                break

            # --- Algorithm 1, line 14 ---
            # Rarest label: fewest remaining examples among unassigned, random tie-break
            counts_present = label_sums[present]
            min_count = counts_present.min()
            tied = present[counts_present == min_count]
            rarest_j = int(rng.choice(tied))

            # --- Algorithm 1, lines 15–33 ---
            # Iterate over ALL unassigned examples carrying the rarest label
            carriers = rem[label_matrix[rem, rarest_j] == 1]
            for chosen in carriers:
                chosen = int(chosen)

                # Subset(s) with largest desired count for this label
                label_desires = desired_per_label[:, rarest_j]
                max_label_desire = label_desires.max()
                top_subsets = np.where(label_desires == max_label_desire)[0]

                if len(top_subsets) == 1:
                    k = int(top_subsets[0])
                else:
                    # Tie-break by largest overall budget, then randomly
                    budgets = split_budget[top_subsets]
                    max_budget = budgets.max()
                    top2 = top_subsets[budgets == max_budget]
                    k = int(rng.choice(top2))

                assignments[chosen] = k
                unassigned.remove(chosen)

                # Update desired counts for all labels this example carries
                desired_per_label[k] -= label_matrix[chosen]
                desired_per_label = np.maximum(desired_per_label, 0)
                split_budget[k] -= 1

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
    
    def combine_metadata(
        self,
        fastsurfer_path: str = "data/metadata/fastsurfer_aggregated_normalized.csv",
        mock_metadata_path: str = "data/metadata/mock_metadata_normalized.csv",
        output_path: str = "data/metadata/metadata_combined.csv",
    ) -> pd.DataFrame:
        fastsurfer = pd.read_csv(fastsurfer_path)
        mock = pd.read_csv(mock_metadata_path)

        fastsurfer["mr_hunt_id"] = fastsurfer["mr_hunt_id"].astype(np.int64)
        mock = mock.rename(columns={"MR_HUNT_ID": "mr_hunt_id"})
        mock["mr_hunt_id"] = mock["mr_hunt_id"].astype(np.int64)

        fastsurfer_feat_cols = [c for c in fastsurfer.columns if c not in ("hunt_id", "mr_hunt_id")]
        mock_feat_cols = [c for c in mock.columns if c != "mr_hunt_id"]
        feat_cols = fastsurfer_feat_cols + mock_feat_cols

        combined = fastsurfer.merge(mock, on="mr_hunt_id", how="outer")

        # Subjects only in mock lack a hunt_id — derive it from mr_hunt_id
        derived_ids = combined["mr_hunt_id"].apply(lambda x: str(int(x))[-5:].zfill(5))
        combined["hunt_id"] = combined["hunt_id"].where(combined["hunt_id"].notna(), derived_ids)

        rows_missing_fastsurfer = int(combined[fastsurfer_feat_cols].isna().any(axis=1).sum())
        rows_missing_mock = int(combined[mock_feat_cols].isna().any(axis=1).sum())
        combined[feat_cols] = combined[feat_cols].fillna(0)

        combined = combined[["hunt_id", "mr_hunt_id"] + feat_cols]
        combined.to_csv(output_path, index=False)

        print(f"[combine_metadata] {len(combined)} rows, {len(combined.columns)} cols → {output_path}")
        if rows_missing_fastsurfer:
            print(f"  {rows_missing_fastsurfer} row(s) missing fastsurfer — filled with 0")
        if rows_missing_mock:
            print(f"  {rows_missing_mock} row(s) missing mock_metadata — filled with 0")
        if not rows_missing_fastsurfer and not rows_missing_mock:
            print("  All rows matched — no gap-filling needed.")

        return combined