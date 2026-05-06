import os
import numpy as np
import pandas as pd
import utils.hunt_id_handler as hih

class HealthDataLoader:
    PID_SAV_NAME = os.path.join("health_data", "NT4MRI-PID.sav")

    # Column classification for ``generate_normalized``. Keyed by source-column name.
    ID_COL = "PID@115827"
    DROP_COLS = [
        "Part@NT2BLQ1", "Part@NT3BLI", "Part@NT3BLM", "Part@NT3BLQ1", "Part@NT3BLQ2",
    ]
    BINARY_COLS = ["Sex", "DiaEv@NT3BLQ1", "WorCu@NT3BLI"]
    CATEGORICAL_COLS = ["Healt@NT3BLQ1", "SmoStat@NT3BLQ1", "Educ@NT2BLQ1"]
    NUMERICAL_COLS = [
        "BirthYear", "SmoPackYrs@NT3BLQ1", "WaistCirc@NT3BLM", "HipCirc@NT3BLM",
        "BPDiasMn23@NT3BLM", "BPSystMn23@NT3BLM", "SeGluNonFast@NT3BLM",
        "Bmi@NT3BLM", "HADSTotExtr@NT3BLQ2",
    ]

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
        self.id_mapper_path       = os.path.join("data", "metadata", "processed", "id_mapping.csv")
        self.out_path: str | None = None
        self._combined            = None
        self._index               = {}

    def sav_to_csv(self, overwrite: bool = False) -> str:
        """Convert ``NT4MRI-PID.sav`` → ``data/metadata/processed/id_mapping.csv``."""
        out_path = self.id_mapper_path
        if os.path.exists(out_path) and not overwrite:
            print(f"[sav_to_csv] {out_path} already exists — skipping (pass overwrite=True to regenerate)")
            return out_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sav_path = os.path.join(self.data_root, self.PID_SAV_NAME)
        df = pd.read_spss(sav_path)
        df.to_csv(out_path, index=False)
        print(f"[sav_to_csv] Saved {len(df)} rows → {out_path}")
        return out_path

    def generate_cleaned(
        self,
        out_dir:            str | None = None,
        health_data_path:   str | None = None,
        overwrite:          bool       = False,
        numerical_strategy: str        = "knn",
        n_neighbors:        int        = 5,
    ) -> str:
        """
        Impute missing values in the raw health-data CSV and write a cleaned
        copy that ``generate_normalized`` can consume.

        Per-column strategy:

        - ``Sex``: NaN → 0 (Female by source-data convention); not "missing".
        - Other binary columns (``BINARY_COLS`` minus ``Sex``): mode imputation.
        - Categorical columns (``CATEGORICAL_COLS``): mode imputation.
        - Numerical columns (``NUMERICAL_COLS``): KNN imputation over all
          numerical columns jointly (default, ``numerical_strategy='knn'``)
          with ``n_neighbors``; or per-column ``'median'`` / ``'mean'``.
        - ``ID_COL`` rows with NaN are dropped with a warning.
        - ``DROP_COLS`` (participation flags) pass through untouched.

        Parameters
        ----------
        out_dir : str, optional
            Directory for the cleaned CSV. Defaults to
            ``<data_root>/health_data``.
        health_data_path : str, optional
            Source CSV. Defaults to ``<data_root>/health_data/health_data.csv``.
        overwrite : bool
            If False (default) and the output already exists, skip and return
            the existing path.
        numerical_strategy : {'knn', 'median', 'mean'}
            Imputation strategy for numerical columns.
        n_neighbors : int
            ``k`` for KNN imputation; ignored unless ``numerical_strategy='knn'``.

        Returns
        -------
        str
            Path to ``<out_dir>/health_data_clean.csv``.
        """
        out_dir          = out_dir          or os.path.join(self.data_root, "health_data")
        health_data_path = health_data_path or os.path.join(self.data_root, "health_data", "health_data.csv")
        out_path         = os.path.join(out_dir, "health_data_clean.csv")

        if os.path.exists(out_path) and not overwrite:
            print(f"[generate_cleaned] {out_path} already exists — skipping (pass overwrite=True to regenerate)")
            return out_path

        os.makedirs(out_dir, exist_ok=True)
        df = pd.read_csv(health_data_path)

        # Sex: NaN means Female by convention — set BEFORE imputation.
        if "Sex" in df.columns:
            n_sex_nan = int(df["Sex"].isna().sum())
            df["Sex"] = df["Sex"].fillna(0).astype(int)
            if n_sex_nan:
                print(f"[generate_cleaned] Sex: {n_sex_nan} NaN → 0 (Female by convention)")

        # Drop rows with missing ID
        if self.ID_COL in df.columns:
            n_id_nan = int(df[self.ID_COL].isna().sum())
            if n_id_nan:
                print(f"[generate_cleaned] {self.ID_COL}: dropping {n_id_nan} rows with missing ID")
                df = df.dropna(subset=[self.ID_COL]).reset_index(drop=True)

        # Mode-impute remaining binary + categorical columns
        mode_cols = [c for c in self.BINARY_COLS + self.CATEGORICAL_COLS
                     if c in df.columns and c != "Sex"]
        n_imputed_total = 0
        cols_imputed    = 0
        for col in mode_cols:
            n_nan = int(df[col].isna().sum())
            if n_nan == 0:
                continue
            mode_val = df[col].mode(dropna=True)
            if mode_val.empty:
                print(f"[generate_cleaned] {col}: all values missing — leaving untouched")
                continue
            df[col] = df[col].fillna(mode_val.iloc[0])
            print(f"[generate_cleaned] {col}: imputed {n_nan} values via mode (={mode_val.iloc[0]!r})")
            n_imputed_total += n_nan
            cols_imputed    += 1

        # Numerical imputation
        num_cols_present = [c for c in self.NUMERICAL_COLS if c in df.columns]
        if num_cols_present:
            n_num_nan_per_col = df[num_cols_present].isna().sum()
            n_num_nan_total   = int(n_num_nan_per_col.sum())
            if n_num_nan_total:
                if numerical_strategy == "knn":
                    from sklearn.impute import KNNImputer
                    imputer = KNNImputer(n_neighbors=n_neighbors)
                    df[num_cols_present] = imputer.fit_transform(df[num_cols_present])
                    label = f"KNN(k={n_neighbors})"
                elif numerical_strategy in ("median", "mean"):
                    fill_vals = (df[num_cols_present].median()
                                 if numerical_strategy == "median"
                                 else df[num_cols_present].mean())
                    df[num_cols_present] = df[num_cols_present].fillna(fill_vals)
                    label = numerical_strategy
                else:
                    raise ValueError(
                        f"numerical_strategy must be 'knn', 'median', or 'mean'; "
                        f"got {numerical_strategy!r}"
                    )
                for col, n in n_num_nan_per_col.items():
                    if n:
                        print(f"[generate_cleaned] {col}: imputed {int(n)} values via {label}")
                n_imputed_total += n_num_nan_total
                cols_imputed    += int((n_num_nan_per_col > 0).sum())

        df.to_csv(out_path, index=False)
        print(f"[generate_cleaned] Imputed {n_imputed_total} values across {cols_imputed} cols → {out_path}")
        return out_path

    def generate_normalized(
        self,
        out_dir: str,
        health_data_path: str | None = None,
        overwrite: bool = False,
    ) -> str:
        """
        Build a normalized health-features CSV from a cleaned HUNT health-data CSV.

        Column types are classified by name (see the class-level ``BINARY_COLS``,
        ``CATEGORICAL_COLS``, ``NUMERICAL_COLS``, ``DROP_COLS`` constants):

        - ``Sex`` is binary with ``1`` = Male and missing/[NONE] = Female (→ 0).
        - Other binary columns are cast to int 0/1.
        - Numerical columns are min-max scaled to [0, 1].
        - Categorical columns are one-hot encoded.
        - ``Part@*`` columns are dropped (participation flags, not features).
        - ``PID@115827`` (long ID) is mapped to ``hunt_id`` (5-char short ID).

        After Sex is normalized (its [NONE] is by design), any remaining missing
        value is reported with its (row, column) and replaced with 0. The input
        CSV is expected to have no missing data.

        Parameters
        ----------
        out_dir : str
            Directory where the normalized CSV will be written.
        health_data_path : str, optional
            Path to the cleaned health-data CSV. Defaults to
            ``<root>/health_data/health_data_clean.csv``.
        overwrite : bool
            If False (default) and the output already exists, skip.

        Returns
        -------
        str
            Path to the normalized CSV.
        """
        health_data_path = health_data_path or os.path.join(
            self.data_root, "health_data", "health_data_clean.csv"
        )

        self.out_path = os.path.join(out_dir, self.data_name_normalized)
        if os.path.exists(self.out_path) and not overwrite:
            print(f"[generate_normalized] {self.out_path} already exists — skipping (pass overwrite=True to regenerate)")
            return self.out_path

        df = pd.read_csv(health_data_path)

        # Drop participation-flag columns
        df = df.drop(columns=[c for c in self.DROP_COLS if c in df.columns])

        # Sex: 1 = Male, anything else (incl. missing → Female by convention) = 0
        if "Sex" in df.columns:
            df["Sex"] = (df["Sex"] == 1).astype(int)

        # Report and zero-fill any remaining missing values
        nan_mask = df.isna()
        if nan_mask.any().any():
            for col in nan_mask.columns:
                for idx in df.index[nan_mask[col]]:
                    print(f"[generate_normalized] Missing value at row {idx}, col {col!r} — set to 0")
            df = df.fillna(0)

        # Map PID@115827 (long ID) → hunt_id (short ID) and drop the source column
        df["hunt_id"] = df[self.ID_COL].apply(lambda x: hih.long_to_short(int(x)))
        df = df.dropna(subset=["hunt_id"]).drop(columns=[self.ID_COL])

        # Binary columns (other than Sex, already handled): coerce to int 0/1
        for col in self.BINARY_COLS:
            if col != "Sex" and col in df.columns:
                df[col] = df[col].astype(int)

        # Numerical: min-max scale to [0, 1]
        for col in self.NUMERICAL_COLS:
            if col in df.columns:
                lo, hi = df[col].min(), df[col].max()
                if hi > lo:
                    df[col] = (df[col] - lo) / (hi - lo)

        # Categorical: one-hot encode
        cat_present = [c for c in self.CATEGORICAL_COLS if c in df.columns]
        if cat_present:
            dummies = pd.get_dummies(df[cat_present], columns=cat_present).astype(np.float32)
            df = df.drop(columns=cat_present)
            df = pd.concat([df, dummies], axis=1)

        # hunt_id first
        df = df[["hunt_id"] + [c for c in df.columns if c != "hunt_id"]]
        df.to_csv(self.out_path, index=False)
        print(f"[generate_normalized] Saved {len(df)} rows → {self.out_path}")

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