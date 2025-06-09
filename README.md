# Evo2-1b Fine-tuning for COVID-19 Variant Analysis

This repository contains code for fine-tuning the Evo2-1b model on COVID-19 spike protein sequences to analyze and classify variants.

## Overview

The project uses NVIDIA's BioNeMo framework to fine-tune the Evo2-1b model on COVID-19 spike protein sequence data. The model is trained to recognize patterns in spike protein sequences that correlate with different COVID-19 variants.

## Features

- Preprocessing of COVID-19 spike protein sequence data
- Fine-tuning of Evo2-1b model with optimized parameters
- Automatic checkpoint detection and training resumption
- Variant classification based on sequence features

## Requirements

- Docker with GPU support
- NVIDIA Container Toolkit
- COVID-19 spike protein sequence data

## Usage

```bash
docker run --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/preprocessed_data:/app/preprocessed_data \
  -v $(pwd)/covid_evo2_model:/app/covid_evo2_model \
  -v $(pwd)/run.sh:/app/run.sh \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/evo2_checkpoint:/app/evo2_checkpoint \
  --shm-size=16g \
  evo2-finetune \
  bash /app/run.sh

