import os
import numpy as np
import pandas as pd


class CombinedMetadataLoader:
    """
    Merges normalized health and FastSurfer metadata into a single CSV, then
    exposes it as a contiguous float32 array for fast per-subject lookups.

    Typical usage
    -------------
    loader = CombinedMetadataLoader(health_root="data/metadata",
                                    fastsurfer_root="data/metadata/hdd/sMRI")
    loader.combine()          # merge + save + load into memory
    vec = loader.get("00039") # O(1) numpy row
    """

    def __init__(
        self,
        output_path: str = "data/metadata/metadata.csv",
    ):
        self.output_path = output_path
        self._feature_names: list[str] = []
        self._features: np.ndarray | None = None   # (N, F) float32
        self._id_to_idx: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def combine(
            self, 
            overwrite: bool = False,
            health_data_path: str = "data/metadata/processed/health_data_normalized.csv",
            fastsurfer_data_path: str = "data/metadata/processed/fastsurfer_data_normalized.csv",
            output_path: str = "data/metadata/metadata.csv"
            ) -> str:
        """
        Merge the two normalized CSVs on ``hunt_id``, save to ``output_path``,
        and load the result into memory.

        If ``output_path`` already exists and ``overwrite=False``, the merge is
        skipped and the existing file is loaded instead.

        Parameters
        ----------
        overwrite : bool
            If True, re-merge and replace the output file even if it exists.

        Returns
        -------
        str
            Path to the combined CSV.
        """
        if os.path.exists(self.output_path) and not overwrite:
            print(f"[combine] {self.output_path} exists — skipping (pass overwrite=True to regenerate)")
            self._load()
            return self.output_path

        health = pd.read_csv(health_data_path)
        health["hunt_id"] = health["hunt_id"].astype(str).str.zfill(5)

        fastsurfer = pd.read_csv(fastsurfer_data_path)
        fastsurfer["hunt_id"] = fastsurfer["hunt_id"].astype(str).str.zfill(5)

        merged = health.merge(fastsurfer, on="hunt_id", how="inner")
        merged.to_csv(self.output_path, index=False)
        print(f"[combine] {len(merged)} subjects, {len(merged.columns)} cols → {self.output_path}")

        self._build_arrays(merged)
        return self.output_path

    def get(self, hunt_id_or_path: str) -> np.ndarray | None:
        """
        Return a float32 feature vector for one subject, or None if unknown.

        Accepts a bare hunt_id or a file path whose basename starts with it
        (e.g. ``'00039_0_T1_PREP_MNI.nii.gz'``).
        """
        self._load()
        idx = self._id_to_idx.get(self._resolve_id(hunt_id_or_path))
        if idx is None:
            print(f"CombinedMetadataLoader: no entry for {hunt_id_or_path!r}")
            return None
        return self._features[idx]

    def get_many(self, ids: list[str]) -> np.ndarray:
        """
        Return a ``(N, F)`` float32 array for a list of IDs or paths.
        Missing subjects are filled with zeros.
        """
        self._load()
        rows = [
            self.get(id_) or np.zeros(self.n_features, dtype=np.float32)
            for id_ in ids
        ]
        return np.stack(rows)

    @property
    def feature_names(self) -> list[str]:
        self._load()
        return list(self._feature_names)

    @property
    def n_features(self) -> int:
        self._load()
        return int(self._features.shape[1])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_arrays(self, df: pd.DataFrame) -> None:
        self._feature_names = [c for c in df.columns if c != "hunt_id"]
        self._features = df[self._feature_names].values.astype(np.float32)
        hunt_ids = df["hunt_id"].astype(str).str.zfill(5)
        self._id_to_idx = {hid: i for i, hid in enumerate(hunt_ids)}

    def _load(self) -> None:
        if self._features is not None:
            return
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(
                f"Combined metadata not found at '{self.output_path}'. Run combine() first."
            )
        self._build_arrays(pd.read_csv(self.output_path))

    def _resolve_id(self, id_or_path: str) -> str:
        if os.sep in id_or_path or "/" in id_or_path:
            return os.path.basename(id_or_path).split("_")[0]
        return str(id_or_path).zfill(5)


class SubsetCombinedMetadataLoader:
    """
    Thin wrapper around CombinedMetadataLoader that exposes only a chosen
    subset of features. Drop-in replacement for CombinedMetadataLoader
    (same .get() / .n_features / .feature_names interface).
    """

    def __init__(self, base: CombinedMetadataLoader, indices: list[int]):
        self._base    = base
        self._indices = np.array(indices, dtype=int)

    def get(self, hunt_id_or_path: str) -> np.ndarray | None:
        full = self._base.get(hunt_id_or_path)
        return full[self._indices] if full is not None else None

    def get_many(self, ids: list[str]) -> np.ndarray:
        full = self._base.get_many(ids)
        return full[:, self._indices]

    @property
    def n_features(self) -> int:
        return len(self._indices)

    @property
    def feature_names(self) -> list[str]:
        names = self._base.feature_names
        return [names[i] for i in self._indices]
