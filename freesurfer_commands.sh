#!/usr/bin/env bash
# Reference FastSurfer / Singularity commands used to produce the FastSurfer
# segmentations for this project. Adjust the bind paths and IDs to your setup.

# Run a single subject through FastSurfer.
singularity exec --nv -e \
  -B $HOME/NTNU-Master-project/fastsurfer/my_mri_data:/data \
  -B $HOME/NTNU-Master-project/fastsurfer/my_fastsurfer_analysis:/analysis \
  -B $HOME/NTNU-Master-project/fastsurfer/my_fs_license.txt:/license.txt \
  ./fastsurfer-gpu.sif \
  /fastsurfer/run_fastsurfer.sh \
    --fs_license /license.txt \
    --t1 /data/4.nii \
    --sid subject4 \
    --sd /analysis \
    --3T \
    --threads 4


# ---------- Batch variant ----------

# Build subject_list.txt with every subject whose id is greater than X.
BASE="/cluster/projects/vc/data/mic/closed/MRI_HUNT/images/images_3D_preprocessed/HUNT3"

find "$BASE" -type f \
  -regextype posix-extended \
  -regex '.*/([^/]+)/\1_0_SEG_3_PREP_MNI\.nii\.gz$' \
  | awk -F/ '
      {
        id=$(NF-1)
        if (id+0 > X) print
      }' \
  | sed "s|^$BASE|/data|" \
  | sort > subjects_list.txt

head -n 3 subjects_list.txt
wc -l subjects_list.txt

# Run every subject listed in subject_list.txt.
singularity exec --nv \
  --no-home \
  -B /cluster/projects/vc/data/mic/closed/MRI_HUNT/images/images_3D_preprocessed/HUNT3:/data:ro \
  -B "$PWD/subjects_list.txt":/subjects_list.txt:ro \
  -B $HOME/NTNU-Master-project/fastsurfer/my_fastsurfer_analysis:/output \
  -B $HOME/NTNU-Master-project/fastsurfer:/fs_license:ro \
  ./fastsurfer-gpu.sif \
  /fastsurfer/brun_fastsurfer.sh \
    --fs_license /fs_license/my_fs_license.txt \
    --sd /output \
    --subject_list /subjects_list.txt \
    --3T \
    --threads 24
