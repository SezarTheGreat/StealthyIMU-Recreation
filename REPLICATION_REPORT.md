# StealthyIMU Replication Project Report

## Project Overview
This repository contains our active replication of the **StealthyIMU** framework (NDSS 2023), which demonstrates how smartphone motion sensors (accelerometers and gyroscopes) can be exploited to steal permission-protected private information from Voice User Interfaces (VUIs) like voice assistants.

This report summarizes our environment setup, the fixes we applied to get the outdated open-source repository functional on modern hardware, and our roadmap for training the models.

---

## 1. Environment & Dataset Resolution
We encountered several breaking issues when initially setting up the original repository on a modern Windows environment without a CUDA GPU. To preserve the integrity of the original codebase, we resolved these using non-destructive wrappers rather than editing the core files:

* **Dataset Pathing:** The dataset was originally hardcoded to specific absolute paths. We migrated the dataset directly into the project root (`StealthyIMU_dataset/`) and dynamically re-generated the internal metadata mapping files (`stealthyIMU_all_relative.csv`) to resolve file-not-found errors across different machines.
* **Torchaudio Backend Bug:** We monkey-patched an incompatibility issue with `torchaudio` on Windows that was causing `speechbrain` to crash during dataset preparation.
* **Evaluation Parsing Bug:** We discovered a bug in the JSON serialization logic within `train.py` (`ast.literal_eval` crashing on invalid semantic predictions). We built a dynamic execution wrapper (`run_training.py`) to safely intercept and handle these failures during evaluation without modifying the original researcher's code.

---

## 2. Phase 1: Baseline Teacher Training
The research paper achieves a highly efficient **2MB** model size using Knowledge Distillation. Before achieving that, we must first train the massive **36MB** "Teacher" models (Phase 1).

We are currently evaluating two Phase 1 configurations:
1. **Run 1 (Baseline Repo Match):** Uses 2 CNN blocks before the BiLSTM (`hparams/open_source.yaml`).
2. **Run 2 (Exact Paper Match):** Uses 1 CNN block before the BiLSTM (`hparams/paper_exact.yaml`).

### Proof-of-Concept Accelerated Run
Training these heavy models on a CPU for the paper's required 30 epochs would take roughly **90 hours**. For the purpose of rapidly proving the end-to-end functionality of our pipeline, we have temporarily dropped the training loops to **3 epochs** (~9 hours). 

*Note: As a result of this intentional underfitting, the interim Word Error Rate (WER) and Semantic Error Rate (SER) will be significantly higher than the 8.5% SER reported in the original paper. The goal right now is purely pipeline validation.*

---

## 3. Phase 2: Knowledge Distillation (Upcoming)
Once the baseline teacher models finish their 3-epoch validation runs, we will proceed to **Phase 2**. 

In Phase 2, we will design and implement the **Teacher-Student Knowledge Distillation** architecture described in the paper. This involves using the probability distributions produced by the large 36MB teacher to train a highly compressed, stealthy 2MB student model capable of running efficiently on a mobile device's background services.
