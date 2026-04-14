# Dataset and Reproduction Notes

## Dataset

The project uses EDNet-KT1 response sequences and EDNet question metadata. Raw EDNet files are not included in this repository. A local copy of the dataset is required.

Expected input files:

```text
/path/to/EdNet-KT1/KT1
/path/to/EdNet-Contents/contents/questions.csv
```

The preprocessing script reads per-student response files, aligns responses with question metadata, and computes a binary correctness label.

## Label Definition

For each valid response event:

```text
label = 1 if user_answer == correct_answer
label = 0 otherwise
```

Rows with missing response or missing question metadata are removed.

## Splitting Strategy

The processed dataset is split by student:

```text
train: 70 percent of students
validation: 15 percent of students
test: 15 percent of students
```

The same student never appears in multiple splits. This makes the evaluation closer to predicting performance for held-out students rather than held-out attempts from students already seen during training.

## Minimal Reproduction

Prepare a small dataset:

```bash
python scripts/prepare_kt1.py \
  --kt1-dir /path/to/EdNet-KT1/KT1 \
  --questions /path/to/EdNet-Contents/contents/questions.csv \
  --out-dir data/processed/kt1_tiny \
  --target-users 300 \
  --max-raw-users 2500 \
  --min-responses 10 \
  --seed 42
```

Run an MLP baseline:

```bash
python scripts/train_model.py \
  --data-dir data/processed/kt1_tiny \
  --run-name tiny_mlp_seq50 \
  --model mlp \
  --seq-len 50 \
  --epochs 3 \
  --batch-size 512
```

Run a GRU sequence model:

```bash
python scripts/train_model.py \
  --data-dir data/processed/kt1_tiny \
  --run-name tiny_gru_seq50 \
  --model gru \
  --seq-len 50 \
  --epochs 3 \
  --batch-size 512
```

For larger experiments, increase `--target-users` and `--max-raw-users`. The reported scale-validation experiments used 2k, 5k, and 10k filtered students.
