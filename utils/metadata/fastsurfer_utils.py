import os
import re
import csv
import glob
import tqdm
import numpy as np
import pandas as pd
import utils.hunt_id_handler as hih

# ---------------------------------------------------------------------------
# Lobe → DK atlas region mapping (based on bash files from St Olavs)
# ---------------------------------------------------------------------------
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

_SUBJ_RE = re.compile(r"FREESURFERRECON_(\d+)_")
_DEFAULT_TEMPLATE = "data/metadata/hdd/sMRI/MRI_9410000000039_all.csv"


# ---------------------------------------------------------------------------
# Stats file parsers
# ---------------------------------------------------------------------------

def _parse_aparc_stats(filepath: str) -> tuple[dict, dict]:
    """
    Parse lh/rh.aparc.stats.

    Returns
    -------
    regions : dict[str, dict]
        {region_name: {"SurfArea": float, "GrayVol": float, "ThickAvg": float}}
    global_measures : dict[str, float]
        Values from '# Measure' header lines (MeanThickness, WhiteSurfArea, etc.).
    """
    regions: dict[str, dict] = {}
    global_measures: dict[str, float] = {}
    try:
        with open(filepath) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("#"):
                    m = re.match(r"# Measure \S+, (\S+), .+, ([\d.]+),", line)
                    if m:
                        try:
                            global_measures[m.group(1)] = float(m.group(2))
                        except ValueError:
                            pass
                elif line:
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            regions[parts[0]] = {
                                "SurfArea": float(parts[2]),
                                "GrayVol":  float(parts[3]),
                                "ThickAvg": float(parts[4]),
                            }
                        except (ValueError, IndexError):
                            pass
    except FileNotFoundError:
        pass
    return regions, global_measures


def _parse_aseg_stats(filepath: str) -> tuple[dict, dict]:
    """
    Parse aseg.stats.

    Returns
    -------
    structures : dict[str, dict]
        {struct_name: {"Volume_mm3": float, "SegId": int, "normMean": float}}
    global_measures : dict[str, float]
        Values from '# Measure' header lines (eTIV, BrainSegVolNotVent, etc.).
    """
    structures: dict[str, dict] = {}
    global_measures: dict[str, float] = {}
    try:
        with open(filepath) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("#"):
                    m = re.match(r"# Measure [\w-]+, ([\w-]+), .+, ([\d.]+),", line)
                    if m:
                        try:
                            global_measures[m.group(1)] = float(m.group(2))
                        except ValueError:
                            pass
                elif line:
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            structures[parts[4]] = {
                                "SegId":      int(parts[1]),
                                "Volume_mm3": float(parts[3]),
                                "normMean":   float(parts[5]),
                            }
                        except (ValueError, IndexError):
                            pass
    except FileNotFoundError:
        pass
    return structures, global_measures


# ---------------------------------------------------------------------------
# CSV generation helpers
# ---------------------------------------------------------------------------

def _put(row: dict, key: str, value: object) -> None:
    """Write value into row[key] only if key exists in row."""
    if key in row:
        row[key] = value


def _sum_structs(structs: dict, names: list[str]) -> float | None:
    """Sum Volume_mm3 for each named structure; returns None if any are missing."""
    total = 0.0
    for name in names:
        if name not in structs:
            return None
        total += structs[name]["Volume_mm3"]
    return total


def _sample_t1w_intensities(subject_dir: str, aseg_structs: dict) -> dict[str, float]:
    """
    Compute mean T1w intensity per subcortical structure by sampling mri/T1.mgz
    at the voxels of each label in mri/aseg.mgz.

    Returns an empty dict if nibabel is not installed or the MRI files are absent.
    """
    try:
        import nibabel as nib
    except ImportError:
        return {}

    t1_path  = os.path.join(subject_dir, "mri", "T1.mgz")
    seg_path = os.path.join(subject_dir, "mri", "aseg.mgz")
    if not (os.path.exists(t1_path) and os.path.exists(seg_path)):
        return {}

    t1_data  = nib.load(t1_path).get_fdata(dtype=np.float32)
    seg_data = np.round(nib.load(seg_path).get_fdata()).astype(np.int32)

    result: dict[str, float] = {}
    for struct, info in aseg_structs.items():
        mask = seg_data == info["SegId"]
        if mask.sum() > 0:
            result[struct] = float(t1_data[mask].mean())

    # White matter is excluded from the aseg data table but expected in the CSV
    wm_labels = {"Left-Cerebral-White-Matter": 2, "Right-Cerebral-White-Matter": 41}
    for struct, seg_id in wm_labels.items():
        if struct not in result:
            mask = seg_data == seg_id
            if mask.sum() > 0:
                result[struct] = float(t1_data[mask].mean())

    return result


# ---------------------------------------------------------------------------
# Public CSV generation functions
# ---------------------------------------------------------------------------

def generate_subject_csv(
    subject_dir: str,
    output_dir: str,
    template_csv: str | None = None,
    use_t1w_sampling: bool = True,
    overwrite: bool = False,
) -> str:
    """
    Generate a per-subject sMRI CSV from a FREESURFERRECON_* directory.

    The output file is named MRI_<long_id>_all.csv and placed in output_dir.
    Its columns match the template CSV exactly (same names, same order).

    Parameters
    ----------
    subject_dir : str
        Path to a FREESURFERRECON_<long_id>_* directory.
    output_dir : str
        Directory where the output CSV will be written.
    template_csv : str, optional
        Reference CSV whose header defines the output columns.
        Defaults to data/metadata/hdd/sMRI/MRI_9410000000039_all.csv.
    use_t1w_sampling : bool
        If True, sample subcort_T1w intensities from mri/T1.mgz (requires nibabel).
    overwrite : bool
        If False and the output CSV already exists, skip writing and return its path.

    Returns
    -------
    str
        Absolute path to the written (or already existing) CSV.
    """
    if template_csv is None:
        template_csv = _DEFAULT_TEMPLATE

    dirname = os.path.basename(subject_dir.rstrip("/\\"))
    m = _SUBJ_RE.match(dirname)
    if not m:
        raise ValueError(f"Cannot parse FREESURFERRECON directory name: {dirname!r}")
    long_id = int(m.group(1))

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"MRI_{long_id}_all.csv")

    if not overwrite and os.path.exists(out_path):
        return os.path.abspath(out_path)

    stats_dir = os.path.join(subject_dir, "stats")
    lh_regions, lh_globals = _parse_aparc_stats(os.path.join(stats_dir, "lh.aparc.stats"))
    rh_regions, rh_globals = _parse_aparc_stats(os.path.join(stats_dir, "rh.aparc.stats"))
    aseg_structs, aseg_globals = _parse_aseg_stats(os.path.join(stats_dir, "aseg.stats"))

    t1w_means: dict[str, float] = {}
    if use_t1w_sampling:
        t1w_means = _sample_t1w_intensities(subject_dir, aseg_structs)

    with open(template_csv) as fh:
        columns = next(csv.reader(fh))

    row: dict[str, object] = {col: "NaN" for col in columns}
    row["SubjID"]  = long_id
    row["VisitID"] = long_id

    # DK atlas: per-region cortical thickness, area, volume
    for hemi, hemi_regions in (("lh", lh_regions), ("rh", rh_regions)):
        for region, vals in hemi_regions.items():
            _put(row, f"cort_thick-ctx-{hemi}-{region}", vals["ThickAvg"])
            _put(row, f"cort_area-ctx-{hemi}-{region}",  vals["SurfArea"])
            _put(row, f"cort_vol-ctx-{hemi}-{region}",   vals["GrayVol"])

    # DK atlas: global mean thickness
    lh_mean_t = lh_globals.get("MeanThickness")
    rh_mean_t = rh_globals.get("MeanThickness")
    if lh_mean_t is not None:
        _put(row, "cort_thick-ctx-lh-mean",  lh_mean_t)
        _put(row, "cort_thick-ctx2_lh_mean", lh_mean_t)
    if rh_mean_t is not None:
        _put(row, "cort_thick-ctx-rh-mean",  rh_mean_t)
        _put(row, "cort_thick-ctx2_rh_mean", rh_mean_t)
    if lh_mean_t is not None and rh_mean_t is not None:
        _put(row, "cort_thick-ctx-mean",  (lh_mean_t + rh_mean_t) / 2)
        _put(row, "cort_thick-ctx2_mean", (lh_mean_t + rh_mean_t) / 2)

    # DK atlas: global white surface area
    lh_surf_area = lh_globals.get("WhiteSurfArea")
    rh_surf_area = rh_globals.get("WhiteSurfArea")
    if lh_surf_area is not None:
        _put(row, "cort_area-ctx-lh-total",  lh_surf_area)
        _put(row, "cort_area-ctx2_lh_total", lh_surf_area)
    if rh_surf_area is not None:
        _put(row, "cort_area-ctx-rh-total",  rh_surf_area)
        _put(row, "cort_area-ctx2_rh_total", rh_surf_area)
    if lh_surf_area is not None and rh_surf_area is not None:
        _put(row, "cort_area-ctx-total",  lh_surf_area + rh_surf_area)
        _put(row, "cort_area-ctx2_total", lh_surf_area + rh_surf_area)

    # DK atlas: total cortical gray volume
    lh_vol_total = sum(v["GrayVol"] for v in lh_regions.values())
    rh_vol_total = sum(v["GrayVol"] for v in rh_regions.values())
    _put(row, "cort_vol-ctx-lh-total",  lh_vol_total)
    _put(row, "cort_vol-ctx2_lh_total", lh_vol_total)
    _put(row, "cort_vol-ctx-rh-total",  rh_vol_total)
    _put(row, "cort_vol-ctx2_rh_total", rh_vol_total)
    _put(row, "cort_vol-ctx-total",     lh_vol_total + rh_vol_total)
    _put(row, "cort_vol-ctx2_total",    lh_vol_total + rh_vol_total)

    # Subcortical: per-structure volume and T1w intensity
    for struct, info in aseg_structs.items():
        _put(row, f"subcort_vol-{struct}", info["Volume_mm3"])
    for struct, intensity in t1w_means.items():
        _put(row, f"subcort_T1w-{struct}", intensity)

    # Subcortical: derived global volumes from aseg header measures
    for measure, col in {
        "eTIV":               "subcort_vol-IntracranialVolume",
        "SupraTentorialVol":  "subcort_vol-SupratentorialVolume",
        "SubCortGrayVol":     "subcort_vol-SubCorticalGrayVolume",
        "BrainSegVolNotVent": "subcort_vol-WholeBrain",
    }.items():
        if measure in aseg_globals:
            _put(row, col, aseg_globals[measure])

    for measure, col in {
        "lhCerebralWhiteMatterVol": "subcort_vol-Left-Cerebral-White-Matter",
        "rhCerebralWhiteMatterVol": "subcort_vol-Right-Cerebral-White-Matter",
        "lhCortexVol":              "subcort_vol-Left-Cerebral-Cortex",
        "rhCortexVol":              "subcort_vol-Right-Cerebral-Cortex",
    }.items():
        if measure in aseg_globals:
            _put(row, col, aseg_globals[measure])

    # Subcortical: composite ventricle volumes
    lat_vent = _sum_structs(aseg_structs, [
        "Left-Lateral-Ventricle", "Right-Lateral-Ventricle",
        "Left-Inf-Lat-Vent",      "Right-Inf-Lat-Vent",
    ])
    if lat_vent is not None:
        _put(row, "subcort_vol-LatVentricles", lat_vent)
        third_fourth = _sum_structs(aseg_structs, ["3rd-Ventricle", "4th-Ventricle"])
        if third_fourth is not None:
            _put(row, "subcort_vol-AllVentricles", lat_vent + third_fourth)

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)

    return os.path.abspath(out_path)


def generate_all_csvs(
    subjects_dir: str = "data/metadata/hdd/full",
    output_dir: str = "data/metadata/hdd/sMRI_generated",
    template_csv: str | None = None,
    use_t1w_sampling: bool = True,
    overwrite: bool = False,
    verbose: bool = False,
) -> list[str]:
    """
    Generate sMRI CSVs for all FREESURFERRECON_* directories in subjects_dir.

    Parameters
    ----------
    subjects_dir : str
        Parent directory containing FREESURFERRECON_* subject directories.
    output_dir : str
        Directory where output CSVs are written.
    template_csv : str, optional
        Reference CSV for column headers.
    use_t1w_sampling : bool
        If True, sample subcort_T1w intensities from mri/T1.mgz (requires nibabel).
    overwrite : bool
        If False, skip subjects whose output CSV already exists.
    verbose : bool
        Print one line per subject showing the action taken.

    Returns
    -------
    list[str]
        Paths to all ready CSVs (written or pre-existing).
    """
    subject_dirs = sorted(
        d for d in glob.glob(os.path.join(subjects_dir, "FREESURFERRECON_*"))
        if os.path.isdir(d)
    )
    if not subject_dirs:
        raise FileNotFoundError(f"No FREESURFERRECON_* directories found in {subjects_dir!r}")

    results = []
    n_written = n_skipped = n_failed = 0

    # Add progress bar
    subject_dirs_bar = tqdm.tqdm(subject_dirs, desc="Processing subjects")

    for subject_dir in subject_dirs_bar:
        name = os.path.basename(subject_dir)
        m = _SUBJ_RE.match(name)
        long_id = m.group(1) if m else None
        expected_path = os.path.join(output_dir, f"MRI_{long_id}_all.csv") if long_id else None

        if not overwrite and expected_path and os.path.exists(expected_path):
            if verbose:
                print(f"  [skip]  {expected_path}")
            results.append(os.path.abspath(expected_path))
            n_skipped += 1
            continue

        try:
            out_path = generate_subject_csv(
                subject_dir=subject_dir,
                output_dir=output_dir,
                template_csv=template_csv,
                use_t1w_sampling=use_t1w_sampling,
                overwrite=overwrite,
            )
            if verbose:
                print(f"  [write] {out_path}")
            results.append(out_path)
            n_written += 1
        except Exception as exc:
            if verbose:
                print(f"  [fail]  {name}: {exc}")
            n_failed += 1

    print(
        f"generate_all_csvs: {n_written} written, {n_skipped} skipped, {n_failed} failed "
        f"({len(subject_dirs)} subjects in {subjects_dir!r})"
    )
    return results


# ---------------------------------------------------------------------------
# FastSurferLoader — aggregate features for model training
# ---------------------------------------------------------------------------

class FastSurferLoader:
    """
    Reads FREESURFERRECON_* directories from ``root`` and reduces the FreeSurfer
    stats to 11 aggregate features per subject:

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
        root: str = "data/metadata/hdd/full",
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
        Aggregate per-subject FreeSurfer stats and save to ``<out_dir>/<data_name>``.

        Parameters
        ----------
        out_dir : str
            Directory where the aggregated CSV will be written.
        in_dir : str, optional
            Directory containing FREESURFERRECON_* subject directories.
            Defaults to ``root`` set at construction time.
        valid_ids : set of str, optional
            Long MR_HUNT_IDs to include; all others are skipped.
        overwrite : bool
            If False and the output file already exists, skip.
        verbose : bool
            Print a warning for each subject that cannot be resolved.
        """
        in_dir = in_dir or self.root
        self.data_path = os.path.join(out_dir, self.data_name)
        if os.path.exists(self.data_path) and not overwrite:
            print(f"[load] {self.data_path} exists — skipping (pass overwrite=True to reprocess)")
            return

        self._df = self._parse_dirs(in_dir=in_dir, valid_ids=valid_ids, verbose=verbose)
        self._df.to_csv(self.data_path)
        self.data_path_normalized = None
        print(f"[load] Saved {len(self._df)} subjects → {self.data_path}")

    def normalize(self, out_dir: str, in_path: str | None = None, overwrite: bool = False) -> str:
        """
        Min-max scale all feature columns to [0, 1] and save to
        ``<out_dir>/<data_name_normalized>``.

        Parameters
        ----------
        out_dir : str
            Directory where the normalized CSV will be written.
        in_path : str, optional
            Path to the aggregated CSV. Defaults to the path from the last load() call.
        overwrite : bool
            If False and the output file already exists, skip.

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
                raise FileNotFoundError(f"Raw data not found at '{in_path}'. Run load() first.")
            self._df = pd.read_csv(in_path, index_col="hunt_id")

        norm = self._df.copy()
        for col in self._df.columns:
            col_min, col_max = self._df[col].min(), self._df[col].max()
            norm[col] = (self._df[col] - col_min) / (col_max - col_min) if col_max > col_min else 0.0

        norm.to_csv(self.data_path_normalized)
        print(f"[normalize] Saved {len(norm)} subjects (normalized) → {self.data_path_normalized}")
        return self.data_path_normalized

    def load_and_normalize(
        self,
        in_dir: str | None = None,
        out_dir: str = "data/metadata/processed",
        valid_ids: set | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> str:
        """
        Read pre-generated MRI_*_all.csv files from ``in_dir``, aggregate
        features, normalize, and save both outputs under ``out_dir``.

        ``in_dir`` must be a directory containing files named
        ``MRI_<hunt_long_id>_all.csv`` — i.e. the ``output_dir`` produced by
        ``generate_all_csvs``.  FREESURFERRECON_* directories are never read
        by this method; use ``load()`` for that.

        Returns the path to the normalized CSV.
        """
        source_dir = in_dir or self.root
        os.makedirs(out_dir, exist_ok=True)
        self.data_path = os.path.join(out_dir, self.data_name)

        if os.path.exists(self.data_path) and not overwrite:
            print(f"[load] {self.data_path} exists — skipping (pass overwrite=True to reprocess)")
            self._df = None
        else:
            self._df = self._parse_csv_dir(source_dir, valid_ids=valid_ids, verbose=verbose)
            self._df.to_csv(self.data_path)
            print(f"[load] Saved {len(self._df)} subjects → {self.data_path}")

        return self.normalize(out_dir, overwrite=overwrite)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse_dirs(self, in_dir: str, valid_ids: set | None, verbose: bool) -> pd.DataFrame:
        dirs = sorted(d for d in glob.glob(os.path.join(in_dir, "FREESURFERRECON_*")) if os.path.isdir(d))
        if not dirs:
            raise FileNotFoundError(f"No FREESURFERRECON_* directories found in {in_dir!r}")

        if valid_ids is not None:
            valid_ids = {str(v) for v in valid_ids}
            dirs_before = len(dirs)
            dirs = [
                d for d in dirs
                if (m := _SUBJ_RE.match(os.path.basename(d))) and m.group(1) in valid_ids
            ]
            print(f"ID filter: {len(dirs)}/{dirs_before} dirs match the {len(valid_ids)} provided IDs.")

        rows = [r for d in dirs if (r := self._process_dir(d, verbose=verbose)) is not None]
        df = pd.DataFrame(rows).set_index("hunt_id")

        before = len(df)
        df = df.dropna()
        if (dropped := before - len(df)):
            print(f"Dropped {dropped} subjects with NaN in aggregate features ({before} → {len(df)} subjects).")

        return df

    def _parse_csv_dir(self, in_dir: str, valid_ids: set | None, verbose: bool) -> pd.DataFrame:
        _id_re = re.compile(r"MRI_(\d+)_all\.csv")
        csv_files = sorted(glob.glob(os.path.join(in_dir, "MRI_*_all.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No MRI_*_all.csv files found in {in_dir!r}")

        if valid_ids is not None:
            valid_ids = {str(v) for v in valid_ids}
            files_before = len(csv_files)
            csv_files = [
                f for f in csv_files
                if (m := _id_re.match(os.path.basename(f))) and m.group(1) in valid_ids
            ]
            print(f"ID filter: {len(csv_files)}/{files_before} CSVs match the {len(valid_ids)} provided IDs.")

        rows = [r for f in csv_files if (r := self._process_csv(f, verbose=verbose)) is not None]
        df = pd.DataFrame(rows).set_index("hunt_id")
        return df

    def _process_csv(self, csv_path: str, verbose: bool = False) -> dict | None:
        """Extract aggregate features from a pre-generated MRI_<id>_all.csv file."""
        m = re.match(r"MRI_(\d+)_all\.csv", os.path.basename(csv_path))
        if not m:
            return None
        long_id = int(m.group(1))

        hunt_id = hih.long_to_short(long_id)
        if hunt_id is None:
            short_candidate = str(long_id).zfill(5)
            if hih.short_to_long(short_candidate) is not None:
                hunt_id = short_candidate
            else:
                if verbose:
                    print(f"Warning: {long_id} not found in HUNT4.xlsx — skipping")
                return None

        try:
            row_data = pd.read_csv(csv_path, nrows=1).iloc[0]
        except Exception as exc:
            if verbose:
                print(f"Warning: could not read {csv_path}: {exc}")
            return None

        result: dict = {
            "hunt_id":    hunt_id,
            "wmh_volume": pd.to_numeric(row_data.get("subcort_vol-WM-hypointensities"), errors="coerce"),
        }

        for lobe, regions in LOBE_REGIONS.items():
            thick_vals, area_vals, vol_vals = [], [], []
            for hemi in ("lh", "rh"):
                for region in regions:
                    thick_vals.append(row_data.get(f"cort_thick-ctx-{hemi}-{region}"))
                    area_vals.append(row_data.get(f"cort_area-ctx-{hemi}-{region}"))
                    vol_vals.append(row_data.get(f"cort_vol-ctx-{hemi}-{region}"))

            thick_s = pd.to_numeric(pd.Series(thick_vals), errors="coerce")
            area_s  = pd.to_numeric(pd.Series(area_vals),  errors="coerce")
            vol_s   = pd.to_numeric(pd.Series(vol_vals),   errors="coerce")

            total_area = area_s.sum(skipna=True)
            weighted_mean = (
                (thick_s * area_s).sum(skipna=True) / total_area
                if total_area > 0 else float("nan")
            )
            result[f"{lobe}_thickness_mean"] = weighted_mean
            result[f"{lobe}_volume_total"]   = vol_s.sum(skipna=False)

        return result

    def _process_dir(self, path: str, verbose: bool = False) -> dict | None:
        dirname = os.path.basename(path)
        m = _SUBJ_RE.match(dirname)
        if not m:
            return None
        long_id = int(m.group(1))

        hunt_id = hih.long_to_short(long_id)
        if hunt_id is None:
            short_candidate = str(long_id).zfill(5)
            if hih.short_to_long(short_candidate) is not None:
                hunt_id = short_candidate
            else:
                if verbose:
                    print(f"Warning: {long_id} not found in HUNT4.xlsx — skipping")
                return None

        stats_dir = os.path.join(path, "stats")
        aseg, _  = _parse_aseg_stats(os.path.join(stats_dir, "aseg.stats"))
        lh, _    = _parse_aparc_stats(os.path.join(stats_dir, "lh.aparc.stats"))
        rh, _    = _parse_aparc_stats(os.path.join(stats_dir, "rh.aparc.stats"))

        result = {
            "hunt_id":    hunt_id,
            "wmh_volume": aseg.get("WM-hypointensities", {}).get("Volume_mm3", float("nan")),
        }

        for lobe, regions in LOBE_REGIONS.items():
            thick_vals = pd.to_numeric(
                pd.Series([lh.get(r, {}).get("ThickAvg") for r in regions] +
                           [rh.get(r, {}).get("ThickAvg") for r in regions]),
                errors="coerce",
            )
            area_vals = pd.to_numeric(
                pd.Series([lh.get(r, {}).get("SurfArea") for r in regions] +
                           [rh.get(r, {}).get("SurfArea") for r in regions]),
                errors="coerce",
            )
            vol_vals = pd.to_numeric(
                pd.Series([lh.get(r, {}).get("GrayVol") for r in regions] +
                           [rh.get(r, {}).get("GrayVol") for r in regions]),
                errors="coerce",
            )
            total_area = area_vals.sum(skipna=True)
            weighted_mean = (
                (thick_vals * area_vals).sum(skipna=True) / total_area
                if total_area > 0 else float("nan")
            )
            result[f"{lobe}_thickness_mean"] = weighted_mean
            result[f"{lobe}_volume_total"]   = vol_vals.sum(skipna=False)

        return result
