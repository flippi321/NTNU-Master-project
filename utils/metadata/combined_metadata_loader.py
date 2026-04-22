import os
import numpy as np
import pandas as pd


class CombinedMetadataLoader:
    """
    Fast, numpy-backed loader for metadata_combined.csv.

    All feature columns are loaded into a contiguous float32 array on first
    access. Subject lookups by hunt_id are O(1) dict lookups; no DataFrame
    overhead on the hot path.

    Usage
    -----
    loader = CombinedMetadataLoader()
    features = loader.get("00039")          # 1-D float32 array, shape (36,)
    features = loader.get("path/00039_…")  # path form also accepted
    batch    = loader.get_many(["00039", "00046"])  # shape (2, 36)
    """

    def __init__(self, csv_path: str = "data/metadata/metadata_combined.csv"):
        self._path = csv_path
        self._feature_names: list[str] = []
        self._features: np.ndarray | None = None   # (N, F) float32
        self._id_to_idx: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self):
        if self._features is not None:
            return
        df = pd.read_csv(self._path)
        key_cols = {"hunt_id", "mr_hunt_id"}
        self._feature_names = [c for c in df.columns if c not in key_cols]
        self._features = df[self._feature_names].values.astype(np.float32)
        hunt_ids = df["hunt_id"].astype(str).str.zfill(5)
        self._id_to_idx = {hid: i for i, hid in enumerate(hunt_ids)}

    def _resolve_id(self, id_or_path: str) -> str:
        """Return the 5-digit hunt_id from either a bare ID or a file path."""
        if os.sep in id_or_path or "/" in id_or_path:
            return os.path.basename(id_or_path).split("_")[0]
        return str(id_or_path).zfill(5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, hunt_id_or_path: str) -> np.ndarray | None:
        """
        Return a float32 feature vector for the subject, or None if unknown.

        Accepts either a bare 5-digit hunt_id string or a file path whose
        basename starts with the hunt_id (e.g. '00039_0_T1_PREP_MNI.nii.gz').
        """
        self._load()
        hunt_id = self._resolve_id(hunt_id_or_path)
        idx = self._id_to_idx.get(hunt_id)
        if idx is None:
            print(f"CombinedMetadataLoader: no entry for hunt_id={hunt_id!r}")
            return None
        return self._features[idx]

    def get_many(self, ids: list[str]) -> np.ndarray:
        """
        Return a (N, F) float32 array for a list of subject IDs or file paths.
        Missing subjects are filled with zeros.
        """
        self._load()
        rows = []
        for id_ in ids:
            result = self.get(id_)
            rows.append(result if result is not None else np.zeros(self.n_features, dtype=np.float32))
        return np.stack(rows)

    @property
    def feature_names(self) -> list[str]:
        self._load()
        return list(self._feature_names)

    @property
    def n_features(self) -> int:
        self._load()
        return int(self._features.shape[1])


class SubsetCombinedMetadataLoader:
    """
    Thin wrapper around CombinedMetadataLoader that exposes only a chosen
    subset of features. Drop-in replacement wherever CombinedMetadataLoader
    is accepted (same .get() / .n_features / .feature_names interface).
    """

    def __init__(self, base: CombinedMetadataLoader, indices: list[int]):
        self._base    = base
        self._indices = np.array(indices, dtype=int)

    def get(self, hunt_id_or_path: str) -> np.ndarray | None:
        full = self._base.get(hunt_id_or_path)
        return full[self._indices] if full is not None else None

    @property
    def n_features(self) -> int:
        return len(self._indices)

    @property
    def feature_names(self) -> list[str]:
        names = self._base.feature_names
        return [names[i] for i in self._indices]
