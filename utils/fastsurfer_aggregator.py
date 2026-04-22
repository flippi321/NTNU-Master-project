import os
import glob
import numpy as np
import pandas as pd
import utils.hunt_id_handler as hih

# Mapping of Fastsufers regions to their corresponding lobes, based on bash files form St Olavs
LOBE_REGIONS = {
    "frontal": [
        "caudalanteriorcingulate", "caudalmiddlefrontal", "lateralorbitofrontal",
        "medialorbitofrontal", "paracentral", "parsopercularis", "parsorbitalis",
        "parstriangularis", "precentral", "rostralanteriorcingulate",
        "rostralmiddlefrontal", "superiorfrontal", "frontalpole",
    ],
    "parietal": [
        "inferiorparietal", "isthmuscingulate", "postcentral", "posteriorcingulate",
        "precuneus", "superiorparietal", "supramarginal",
    ],
    "temporal": [
        "bankssts", "entorhinal", "fusiform", "inferiortemporal", "middletemporal",
        "parahippocampal", "superiortemporal", "temporalpole", "transversetemporal",
    ],
    "occipital": [
        "cuneus", "lateraloccipital", "lingual", "pericalcarine",
    ],
    "insula": [
        "insula",
    ],
}
HEMIS = ("lh", "rh")


def _thick_cols(regions):
    return [f"cort_thick-ctx-{h}-{r}" for h in HEMIS for r in regions]

def _area_cols(regions):
    return [f"cort_area-ctx-{h}-{r}" for h in HEMIS for r in regions]

def _vol_cols(regions):
    return [f"cort_vol-ctx-{h}-{r}" for h in HEMIS for r in regions]

class FastSurferAggregator:
    """
    Loads per-subject FastSurfer CSVs from ``smri_dir`` and reduces
    the ~2700-column output to 11 aggregate features per subject:

    - ``wmh_volume``              : total WM-hypointensity volume (mm³)
    - ``{lobe}_thickness_mean``   : surface-area-weighted mean cortical thickness
                                    across all DK parcels in that lobe (both
                                    hemispheres), in mm — matches aparcstats2table
    - ``{lobe}_volume_total``     : total cortical gray-matter volume for that
                                    lobe (both hemispheres), in mm³

    The five lobes are: frontal, parietal, temporal, occipital, insula.
    Insula is kept separate to match FreeSurfer's lobesStrict annotation.
    The DK-atlas parcel→lobe mapping is defined in ``LOBE_REGIONS``.

    Subjects with any remaining NaN in the 9 output columns after
    aggregation are dropped and reported.
    """

    def __init__(self, smri_dir: str = "data/metadata/hdd/sMRI", hunt4_path: str | None = None):
        self.smri_dir = smri_dir
        if hunt4_path is not None:
            hih.init(hunt4_path)
        self._df: pd.DataFrame | None = None

    def load(self, valid_ids: set | None = None, force: bool = False) -> pd.DataFrame:
        """
        Parse all ``*_all.csv`` files and return a tidy DataFrame indexed
        by ``hunt_id`` (HUNT4 MRI Participant number from HUNT4.xlsx).

        Parameters
        ----------
        valid_ids : set of str, optional
            Full MR_HUNT_IDs (e.g. ``{'9410000000039', ...}``).  When given,
            only subjects whose ID appears in the set are processed; files for
            all other subjects are skipped.  Use this to restrict to subjects
            that also have paired MRI data (HUNT3 ∩ HUNT4).
        force : bool
            Re-parse from disk even if a cached result exists.
        """
        if self._df is not None and not force:
            return self._df

        files = sorted(glob.glob(os.path.join(self.smri_dir, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {self.smri_dir!r}")

        if valid_ids is not None:
            valid_ids = {str(v) for v in valid_ids}
            files_before = len(files)
            files = [f for f in files
                     if os.path.basename(f).split("_")[1] in valid_ids]
            print(f"ID filter: {len(files)}/{files_before} files match "
                  f"the {len(valid_ids)} provided IDs.")

        rows = [r for f in files if (r := self._process_file(f)) is not None]
        df = pd.DataFrame(rows).set_index("hunt_id")

        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        if dropped:
            print(f"Dropped {dropped} subjects with NaN in aggregate features "
                  f"({before} → {len(df)} subjects).")

        self._df = df
        return df

    def normalize(self) -> pd.DataFrame:
        """
        Min-max scale all 11 feature columns to [0, 1].

        ``mr_hunt_id`` is passed through unchanged.  Scaling parameters are
        derived from the loaded cohort (same convention as
        ``metadata_normalizer.normalize``).

        Returns
        -------
        pd.DataFrame with the same index and columns as ``load()``, but with
        all feature columns in [0, 1].
        """
        df = self.load()
        feat_cols = [c for c in df.columns if c != "mr_hunt_id"]
        norm = df.copy()
        for col in feat_cols:
            col_min, col_max = df[col].min(), df[col].max()
            norm[col] = (df[col] - col_min) / (col_max - col_min) if col_max > col_min else 0.0
        return norm

    def save(self, out_path: str) -> None:
        """Write the aggregated DataFrame to ``out_path`` as CSV."""
        df = self.load()
        df.to_csv(out_path)
        print(f"Saved {len(df)} subjects → {out_path}")

    def save_normalized(self, out_path: str) -> None:
        """Write the min-max normalized DataFrame to ``out_path`` as CSV."""
        df = self.normalize()
        df.to_csv(out_path)
        print(f"Saved {len(df)} subjects (normalized) → {out_path}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _process_file(self, path: str) -> dict:
        df = pd.read_csv(path)

        # The raw files store missing values as the string "NaN " (with a
        # trailing space) rather than a true float NaN.  Coerce everything
        # numeric so we get real NaNs that pandas can handle.
        df = df.apply(pd.to_numeric, errors="coerce")

        row = df.iloc[0]
        subj_id = int(row["SubjID"])
        hunt_id = hih.long_to_short(subj_id)
        if hunt_id is None:
            print(f"Warning: mr_hunt_id {subj_id} not found in HUNT4.xlsx — skipping")
            return None

        result = {
            "hunt_id":    hunt_id,
            "mr_hunt_id": str(subj_id),
            "wmh_volume": row["subcort_vol-WM-hypointensities"],
        }

        for lobe, regions in LOBE_REGIONS.items():
            thick_vals = pd.to_numeric(
                pd.Series([row.get(c) for c in _thick_cols(regions)]),
                errors="coerce",
            )
            area_vals = pd.to_numeric(
                pd.Series([row.get(c) for c in _area_cols(regions)]),
                errors="coerce",
            )
            vol_vals = pd.to_numeric(
                pd.Series([row.get(c) for c in _vol_cols(regions)]),
                errors="coerce",
            )
            # Area-weighted mean matches aparcstats2table --meas thickness --parc lobe
            total_area = area_vals.sum(skipna=True)
            if total_area > 0:
                weighted_mean = (thick_vals * area_vals).sum(skipna=True) / total_area
            else:
                weighted_mean = float("nan")
            result[f"{lobe}_thickness_mean"] = weighted_mean
            result[f"{lobe}_volume_total"]   = vol_vals.sum(skipna=False)

        return result
