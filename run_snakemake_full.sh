#!/bin/bash

# Run Snakemake with cluster submission
snakemake -s full_pipeline_snakefile all \
  --configfile="configs/scDINO_full_pipeline.yaml" \
  --keep-incomplete \
  --drop-metadata \
  --cores 8 \
  --jobs 40 \
  -k \
  --cluster "sbatch --time=03:30:00 \
             --gres=gpu:rtx4090:1 \
             --cpus-per-task=16 \
             --mem=80GB \
             --output=/home/nbahou/myimaging/scDINO/slurm_logs/slurm_output_%j.txt \
             --error=/home/nbahou/myimaging/scDINO/slurm_logs/slurm_error_%j.txt" \
  --latency-wait 90 \
  --rerun-incomplete \
  --forceall