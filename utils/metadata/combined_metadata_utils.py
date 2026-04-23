import os
import numpy as np
import pandas as pd
import utils.hunt_id_handler as hih


class CombinedMetadataUtils:
    """
    Merges normalized health and FastSurfer metadata into a single CSV, then
    exposes it as a contiguous float32 array for fast per-subject lookups.

    Typical usage
    -------------
    loader = CombinedMetadataUtil(health_root="data/metadata",
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

    def check_overlap(
        self,
        health_data_path: str = "data/metadata/processed/health_data_normalized.csv",
        fastsurfer_data_path: str = "data/metadata/processed/fastsurfer_data_normalized.csv",
    ) -> tuple[list[str], list[str]]:
        """
        Compare the hunt_ids present in each source file and report coverage.

        Prints a summary table and returns the long IDs of subjects that are
        missing from each side.

        Parameters
        ----------
        health_data_path : str
            Path to the normalized health data CSV.
        fastsurfer_data_path : str
            Path to the normalized FastSurfer data CSV.

        Returns
        -------
        health_only_long : list of str
            Long MR_HUNT_IDs present in health data but not FastSurfer.
        fastsurfer_only_long : list of str
            Long MR_HUNT_IDs present in FastSurfer but not health data.
        """
        health_ids      = set(pd.read_csv(health_data_path,     usecols=["hunt_id"])["hunt_id"].astype(str).str.zfill(5))
        fastsurfer_ids  = set(pd.read_csv(fastsurfer_data_path, usecols=["hunt_id"])["hunt_id"].astype(str).str.zfill(5))

        overlap         = health_ids & fastsurfer_ids
        health_only     = sorted(health_ids    - fastsurfer_ids)
        fastsurfer_only = sorted(fastsurfer_ids - health_ids)

        print(f"Health data:      {len(health_ids)} subjects")
        print(f"FastSurfer data:  {len(fastsurfer_ids)} subjects")
        print(f"Overlap:          {len(overlap)} subjects")
        print(f"Health only:      {len(health_only)} subjects")
        print(f"FastSurfer only:  {len(fastsurfer_only)} subjects")

        def _to_long(ids):
            return [str(hih.short_to_long(i) or "—") for i in ids]

        health_only_long     = _to_long(health_only)
        fastsurfer_only_long = _to_long(fastsurfer_only)

        for label, short_ids, long_ids in [
            ("Health only",    health_only,     health_only_long),
            ("FastSurfer only", fastsurfer_only, fastsurfer_only_long),
        ]:
            if not short_ids:
                continue
            print(f"\n{label}:")
            print(f"  {'hunt_id':<10}  long_id")
            print(f"  {'-'*9}  {'-'*15}")
            for s, l in zip(short_ids, long_ids):
                print(f"  {s:<10}  {l}")

        return health_only_long, fastsurfer_only_long

    def get(self, hunt_id_or_path: str) -> np.ndarray | None:
        """
        Return a float32 feature vector for one subject, or None if unknown.

        Accepts a bare hunt_id or a file path whose basename starts with it
        (e.g. ``'00039_0_T1_PREP_MNI.nii.gz'``).
        """
        self._load()
        idx = self._id_to_idx.get(self._resolve_id(hunt_id_or_path))
        if idx is None:
            print(f"CombinedMetadataUtil: no entry for {hunt_id_or_path!r}")
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


class SubsetCombinedMetadataUtil:
    """
    Thin wrapper around CombinedMetadataUtil that exposes only a chosen
    subset of features. Drop-in replacement for CombinedMetadataUtil
    (same .get() / .n_features / .feature_names interface).
    """

    def __init__(self, base: CombinedMetadataUtil, indices: list[int]):
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
