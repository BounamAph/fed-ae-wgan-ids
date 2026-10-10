# Federated learning with AE-WGAN for network intrusion detection

Intrusion detection models miss **rare attacks** (remote-to-local, user-to-root) because they barely appear in the training data. This project **reproduces** a 2025 paper that generates synthetic attacks with a Denoising Autoencoder + Wasserstein GAN (AE-WGAN) to rebalance the classes, then **extends it with federated learning** (FedAvg), so that several sites can train a shared detector without ever sharing their raw traffic.

*Final project, COMP 542 Machine Learning, California State University, Northridge (CSUN), spring 2026. Grade: A.*

## Key results

| | Result |
|---|---|
| Rare attacks (R2L), random forest, balanced dataset | F1 **0.080 → 0.365** (+28.5 points; the paper reports +37) |
| Exact feature set of the paper (24 features, Table 5) | binary F1 **0.79 → 0.91**: preprocessing mattered more than the model |
| Federated learning (FedAvg, 4 non-IID clients, 5 rounds) | within **0.2 to 0.4 F1 points** of centralised training (RF 0.862 vs 0.866, CNN 0.907 vs 0.909) |
| CICIDS-2017, multi-class, balanced | CNN **0.685** (the paper reports 0.65) |

### Paper vs published code

While debugging six successive versions, I found that the authors' public code differs from the paper: it is a BiGAN (the discriminator also receives the latent vector), the generator has 5 layers instead of 3, the critic is trained 1:1 instead of 5:1, and it generates N + 20,000 samples instead of the 10% described. Notebook `V3a` reproduces the published architecture to measure the impact. Details are in the [report](report/report.pdf).

## Repository

```
code/          modular Python pipeline (main.py + preprocessing, AE-WGAN, generative scores, classifiers)
notebooks/
  NSL-KDD/             V1 → V6, one notebook per iteration (executed_with_outputs/: V2 and V3b with cell outputs)
  Federated_Learning/  FedAvg + AE-WGAN
  CICIDS-2017/         C1 → C3
results/       one folder per experiment (CSV + chart)
report/        final report (7 pages)
```

| Version | Main change | Best binary F1 |
|---|---|---|
| V1 | first implementation (WGAN-GP) | 0.791 |
| V2 | deeper generator, Adamax | 0.805 |
| V3 | authors' published architecture (BiGAN) | 0.792 |
| V4 | oversampling to 10% of the Normal class | 0.800 |
| **V5** | **exact 24 features of the paper** | **0.912** |
| V5b, V5c | per-category F1, R2L-specific training | |
| V6 | balanced dataset (70,343 samples) | evaluated per attack category |

Six classifiers are compared: decision tree, random forest, CNN, GRU, LSTM and RNN (DT, RF, CNN and GRU on CICIDS-2017), in binary and multi-class settings.

## Run

The notebooks run on **Google Colab**: NSL-KDD is downloaded automatically; CICIDS-2017 must be placed in your Google Drive (`/content/drive/MyDrive/cicids2017/...`).

The Python pipeline runs locally (TensorFlow / Keras, scikit-learn, pandas):

```bash
cd code
python main.py --dataset nslkdd --train_path KDDTrain+.txt --test_path KDDTest+.txt --output_dir results
```

Datasets are not included in this repository (see Credits for where to get them).

## Limits

- NSL-KDD dates from 2009: good for comparing methods, far from today's traffic.
- U2R attacks remain hard to detect; the LSTM was unstable with synthetic data.
- Federated learning was tested on NSL-KDD only, with simulated clients.

## How it was built

I designed the study (reproduction, federated extension, the six iterations and their analysis) and developed the code assisted by **Claude**, an AI coding assistant.

## Credits

This work builds on the following, which I gratefully acknowledge:

- **Reference paper (reproduced):** M. Arafah, I. Phillips, A. Adnane, W. Hadi, M. Alauthman and A.-K. Al-Banna, "Anomaly-based network intrusion detection using denoising autoencoder and Wasserstein GAN synthetic attacks", *Applied Soft Computing*, vol. 168, 112455, 2025. [doi:10.1016/j.asoc.2024.112455](https://doi.org/10.1016/j.asoc.2024.112455)
- **Authors' code:** [github.com/marafah/AE-WGAN](https://github.com/marafah/AE-WGAN). Notebook `V3a` re-implements the architecture published there to compare it with the paper. The original code is not redistributed here.
- **Federated learning:** B. Buyuktanir, Ş. Altinkaya, G. Karatas Baydogmus and K. Yildiz, "Federated learning in intrusion detection: advancements, applications, and future directions", *Cluster Computing*, vol. 28, 473, 2025. [doi:10.1007/s10586-025-05325-w](https://doi.org/10.1007/s10586-025-05325-w). FedAvg was introduced by McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017.
- **NSL-KDD dataset:** M. Tavallaee, E. Bagheri, W. Lu and A. A. Ghorbani, "A detailed analysis of the KDD CUP 99 data set", IEEE CISDA 2009. Canadian Institute for Cybersecurity, University of New Brunswick. Downloaded from the [defcom17/NSL_KDD](https://github.com/defcom17/NSL_KDD) mirror.
- **CICIDS-2017 dataset:** I. Sharafaldin, A. H. Lashkari and A. A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", ICISSP 2018. Canadian Institute for Cybersecurity, University of New Brunswick.
- **Course:** COMP 542 Machine Learning, California State University, Northridge.

## Author

**Bounam Aphayavong**, engineering student in cybersecurity at ENSIIE (Paris-Saclay / IMT), exchange semester at CSUN.
