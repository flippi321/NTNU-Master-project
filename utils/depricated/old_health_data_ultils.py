import os
import numpy as np
import pandas as pd
import utils.hunt_id_handler as hih

class HealthDataLoader:
    def __init__(
        self,
        root: str = "data/metadata",
        data_name_normalized: str = "health_data_normalized.csv",
    ):
        """
        Parameters
        ----------
        root : str
            Directory containing HUNT3.csv and mock_metadata.csv.
        data_name_normalized : str
            Filename for the normalized output CSV.
        """
        self.data_root            = root
        self.data_name_normalized = data_name_normalized
        self.out_path: str | None = None
        self._combined            = None
        self._index               = {}

    def generate_normalized(
        self,
        out_dir: str,
        hunt3_path: str | None = None,
        health_features_path: str | None = None,
        overwrite: bool = False,
    ) -> str:
        """
        Build a normalized health metadata CSV from HUNT3 and mock_metadata.

        Uses ``hih.long_to_short()`` to map MR_HUNT_IDs to 5-digit hunt_ids.
        All numeric feature columns are min-max scaled to [0, 1].
        Writes to ``<out_dir>/<data_name_normalized>``.

        Parameters
        ----------
        out_dir : str
            Directory where the normalized CSV will be written.
        hunt3_path : str, optional
            Path to HUNT3.csv. Defaults to ``<root>/HUNT3.csv``.
        health_features_path : str, optional
            Path to mock_metadata.csv containing per-subject health features.
            Defaults to ``<root>/mock_metadata.csv``.
        overwrite : bool
            If False (default) and the file already exists, skip.

        Returns
        -------
        str
            Path to the normalized CSV.
        """
        hunt3_path           = hunt3_path or os.path.join(self.data_root, "HUNT3.csv")
        health_features_path = health_features_path or os.path.join(self.data_root, "mock_metadata.csv")

        self.out_path = os.path.join(out_dir, self.data_name_normalized)
        if os.path.exists(self.out_path) and not overwrite:
            print(f"[generate_normalized] {self.out_path} already exists — skipping (pass overwrite=True to regenerate)")
            return self.out_path

        hunt3 = pd.read_csv(hunt3_path)
        hunt3 = hunt3.rename(columns={"MR_HUNT_ID": "long_id", "Age_at_time_of_MRI": "age"})
        hunt3["hunt_id"] = hunt3["long_id"].apply(lambda x: hih.long_to_short(int(x)))
        hunt3 = hunt3.dropna(subset=["hunt_id"])

        features = pd.read_csv(health_features_path)
        features = features.rename(columns={"MR_HUNT_ID": "long_id"})

        combined = hunt3[["hunt_id", "long_id", "age"]].merge(
            features, on="long_id", how="inner"
        ).drop(columns=["long_id"])

        id_cols = {"hunt_id"}
        feat_cols = [c for c in combined.columns if c not in id_cols]
        numeric_cols     = [c for c in feat_cols if     np.issubdtype(combined[c].dtype, np.number)]
        categorical_cols = [c for c in feat_cols if not np.issubdtype(combined[c].dtype, np.number)]

        for col in numeric_cols:
            col_min, col_max = combined[col].min(), combined[col].max()
            if col_max > col_min:
                combined[col] = (combined[col] - col_min) / (col_max - col_min)

        if categorical_cols:
            dummies = pd.get_dummies(combined[categorical_cols]).astype(np.float32)
            combined = combined.drop(columns=categorical_cols)
            combined = pd.concat([combined, dummies], axis=1)

        combined = combined[["hunt_id"] + [c for c in combined.columns if c != "hunt_id"]]
        combined.to_csv(self.out_path, index=False)
        print(f"[generate_normalized] Saved {len(combined)} rows → {self.out_path}")

        self._combined = None
        self._index    = {}
        return self.out_path

    def _get_id_from_path(self, path: str) -> str:
        filename = os.path.basename(path)
        return filename.split('_')[0]

    def _load(self):
        if self._combined is not None:
            return

        if not os.path.exists(self.out_path):
            raise FileNotFoundError(
                f"Normalized data not found at '{self.out_path}'. "
                "Run generate_normalized() first."
            )

        self._combined = pd.read_csv(self.out_path)
        self._index    = self._combined.set_index("hunt_id").to_dict(orient="index")

    def get(self, hunt_path, columns: list[str] | None = None, labeled: bool = False):
        """
        Return feature values for one subject.

        Parameters
        ----------
        hunt_path : str
            File path or bare hunt_id string.
        columns : list of str, optional
            Specific column names to return. Defaults to all feature columns
            (everything except hunt_id and mr_hunt_id).
        labeled : bool
            If True return a dict, otherwise a list.
        """
        self._load()
        hunt_id = self._get_id_from_path(hunt_path)
        row = self._index.get(hunt_id)
        if row is None:
            print(f"No metadata found for hunt_id: {hunt_id}")
            return None
        skip = {"mr_hunt_id"}
        keys = columns if columns is not None else [k for k in row if k not in skip]
        result = {k: row[k] for k in keys}
        return result if labeled else list(result.values())

    def get_many(self, hunt_paths, columns: list[str] | None = None):
        return [self.get(p, columns=columns) for p in hunt_paths]