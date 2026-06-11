# CSE 144 Final Project — Transfer Learning Challenge (Spring 2026)

Fine-tuning an ImageNet-pretrained EfficientNet-B3 to classify a 100-class dataset with ~10 training images per class.

**Kaggle public leaderboard score: 0.763** (baseline: 0.60)

![Leaderboard](leaderboard.png)

## Repository contents

- `CSE144_Final_Project.ipynb` — full training + inference notebook (Google Colab)
- `CSE144_Final_Report.pdf` — project report
- `leaderboard.png` — Kaggle leaderboard screenshot
- Trained model weights (`best_effb3_dropout.pth`, ~50 MB): **[Google Drive link](PASTE_YOUR_DRIVE_LINK_HERE)**

## Approach (summary)

- Backbone: EfficientNet-B3, ImageNet pretrained (torchvision)
- Custom classifier head: Dropout(0.4) → Linear(1536, 512) → SiLU → Dropout(0.3) → Linear(512, 100)
- Full fine-tuning with discriminative learning rates (backbone 1e-4, head 1e-3), AdamW, weight decay 1e-4
- Cross-entropy with label smoothing 0.1, cosine annealing over 30 epochs, batch size 32, 300×300 inputs
- Augmentation: RandomResizedCrop, horizontal flip, rotation (±15°), color jitter
- Seed 42 everywhere; 80/20 train/val split; best validation accuracy 68.98%

See the report for the full experimental setup and ablations (EfficientNet-B0 baseline scored 0.627).

## How to run

### Environment

Google Colab with a GPU runtime (tested on an NVIDIA Tesla T4). Uses Colab's default PyTorch/torchvision — no extra installs needed.

### Training from scratch

1. Open `CSE144_Final_Project.ipynb` in Google Colab (Runtime → Change runtime type → GPU).
2. Download the competition data zip from the Kaggle page and upload it to `/content/` as `ucsc-cse-144-spring-2026-final-project.zip`.
3. Run all cells top to bottom. Training takes ~30–40 minutes on a T4.
4. The best checkpoint is saved to `/content/best_effb3_dropout.pth` and the submission file to `/content/submission_effb3_dropout.csv`.

### Inference only (using the provided weights)

1. Complete steps 1–2 above, and run the setup/data/model-definition cells (everything **before** the training loop cell). Skip the training loop.
2. Download `best_effb3_dropout.pth` from the Google Drive link above and upload it to `/content/` in Colab.
3. Run the remaining cells from `model.load_state_dict(torch.load("/content/best_effb3_dropout.pth"))` onward.
4. This regenerates `/content/submission_effb3_dropout.csv`, reproducing the 0.763 public-leaderboard submission.

### Notes on reproducibility

Seeds are fixed (seed 42 for `random`, NumPy, and PyTorch CPU/CUDA), and the train/val split uses a seeded generator. Minor run-to-run variation in training is possible since cuDNN deterministic mode is not enforced, but results should match within ~1–2%.
