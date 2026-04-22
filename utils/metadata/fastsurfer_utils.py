import os
import glob
import numpy as np
import pandas as pd
import utils.hunt_id_handler as hih

# Mapping of FastSurfer regions to their corresponding lobes, based on bash files from St Olavs
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


class FastSurferLoader:
    """
    Loads per-subject FastSurfer CSVs from ``root`` and reduces the ~2700-column
    output to 11 aggregate features per subject:

    - ``wmh_volume``              : total WM-hypointensity volume (mm³)
    - ``{lobe}_thickness_mean``   : surface-area-weighted mean cortical thickness
                                    across all DK parcels in that lobe (both
                                    hemispheres), in mm
    - ``{lobe}_volume_total``     : total cortical gray-matter volume for that
                                    lobe (both hemispheres), in mm³

    The five lobes are: frontal, parietal, temporal, occipital, insula.
    """

    def __init__(
        self,
        root: str = "data/metadata/hdd/sMRI",
        data_name: str = "fastsurfer_data.csv",
        data_name_normalized: str = "fastsurfer_data_normalized.csv",
    ):
        self.root = root
        self.data_name = data_name
        self.data_name_normalized = data_name_normalized
        self.data_path: str | None = None
        self.data_path_normalized: str | None = None
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        out_dir: str,
        in_dir: str | None = None,
        valid_ids: set | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Aggregate per-subject CSVs and save to ``<out_dir>/<data_name>``.

        If the file already exists and ``overwrite=False``, the step is
        skipped. Pass ``overwrite=True`` to reparse from raw CSVs and replace.

        Parameters
        ----------
        out_dir : str
            Directory where the aggregated CSV will be written.
        in_dir : str, optional
            Directory containing the per-subject ``*_all.csv`` files.
            Defaults to ``root`` set at construction time.
        valid_ids : set of str, optional
            Full MR_HUNT_IDs to include; others are skipped.
        overwrite : bool
            If False (default) and the file exists, skip.
        verbose : bool
            Print per-file warnings during processing.
        """
        in_dir = in_dir or self.root
        self.data_path = os.path.join(out_dir, self.data_name)
        if os.path.exists(self.data_path) and not overwrite:
            print(f"[load] {self.data_path} exists — skipping (pass overwrite=True to reprocess)")
            return

        self._df = self._parse_csvs(in_dir=in_dir, valid_ids=valid_ids, verbose=verbose)
        self._df.to_csv(self.data_path)
        self.data_path_normalized = None
        print(f"[load] Saved {len(self._df)} subjects → {self.data_path}")

    def normalize(self, out_dir: str, in_path: str | None = None, overwrite: bool = False) -> str:
        """
        Min-max scale all feature columns to [0, 1] and save to
        ``<out_dir>/<data_name_normalized>``.

        Reads from ``in_path`` if given, otherwise from the path set by the
        last ``load()`` call. ``mr_hunt_id`` is passed through unchanged.

        Parameters
        ----------
        out_dir : str
            Directory where the normalized CSV will be written.
        in_path : str, optional
            Path to the aggregated (un-normalized) CSV. Defaults to the path
            written by the last ``load()`` call.
        overwrite : bool
            If False (default) and the file exists, skip.

        Returns
        -------
        str
            Path to the normalized CSV.
        """
        in_path = in_path or self.data_path
        self.data_path_normalized = os.path.join(out_dir, self.data_name_normalized)
        if os.path.exists(self.data_path_normalized) and not overwrite:
            print(f"[normalize] {self.data_path_normalized} exists — skipping (pass overwrite=True to renormalize)")
            return self.data_path_normalized

        if self._df is None:
            if not in_path or not os.path.exists(in_path):
                raise FileNotFoundError(
                    f"Raw data not found at '{in_path}'. Run load() first."
                )
            self._df = pd.read_csv(in_path, index_col="hunt_id")

        feat_cols = list(self._df.columns)
        norm = self._df.copy()
        for col in feat_cols:
            col_min, col_max = self._df[col].min(), self._df[col].max()
            norm[col] = (self._df[col] - col_min) / (col_max - col_min) if col_max > col_min else 0.0

        norm.to_csv(self.data_path_normalized)
        print(f"[normalize] Saved {len(norm)} subjects (normalized) → {self.data_path_normalized}")
        return self.data_path_normalized

    def load_and_normalize(
        self,
        out_dir: str,
        valid_ids: set | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> str:
        """
        Aggregate and normalize in one call.

        Each step checks its own output file independently, so a partially-complete
        run (raw saved but normalized missing) only redoes the missing step.

        Parameters
        ----------
        out_dir : str
            Directory where both CSVs will be written.
        valid_ids : set of str, optional
            Passed through to ``load()``.
        overwrite : bool
            Passed to both steps.
        verbose : bool
            Passed through to ``load()``.

        Returns
        -------
        str
            Path to the normalized CSV.
        """
        self.load(out_dir, valid_ids=valid_ids, overwrite=overwrite, verbose=verbose)
        return self.normalize(out_dir, overwrite=overwrite)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse_csvs(self, in_dir: str, valid_ids: set | None, verbose: bool) -> pd.DataFrame:
        files = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {in_dir!r}")

        if valid_ids is not None:
            valid_ids = {str(v) for v in valid_ids}
            files_before = len(files)
            files = [f for f in files if os.path.basename(f).split("_")[1] in valid_ids]
            print(f"ID filter: {len(files)}/{files_before} files match the {len(valid_ids)} provided IDs.")

        rows = [r for f in files if (r := self._process_file(f, verbose=verbose)) is not None]
        df = pd.DataFrame(rows).set_index("hunt_id")

        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        if dropped:
            print(f"Dropped {dropped} subjects with NaN in aggregate features ({before} → {len(df)} subjects).")

        return df

    def _process_file(self, path: str, verbose: bool = False) -> dict:
        df = pd.read_csv(path)
        df = df.apply(pd.to_numeric, errors="coerce")

        row = df.iloc[0]
        subj_id = int(row["SubjID"])

        hunt_id = hih.long_to_short(subj_id)
        if hunt_id is None:
            short_candidate = str(subj_id).zfill(5)
            if hih.short_to_long(short_candidate) is not None:
                hunt_id = short_candidate
            else:
                if verbose:
                    print(f"Warning: SubjID {subj_id} not found in HUNT4.xlsx — skipping")
                return None

        result = {
            "hunt_id":    hunt_id,
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
            total_area = area_vals.sum(skipna=True)
            if total_area > 0:
                weighted_mean = (thick_vals * area_vals).sum(skipna=True) / total_area
            else:
                weighted_mean = float("nan")
            result[f"{lobe}_thickness_mean"] = weighted_mean
            result[f"{lobe}_volume_total"]   = vol_vals.sum(skipna=False)

        return result
