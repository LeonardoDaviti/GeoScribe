# %% [markdown]
# # GeoScribe: let's train a Georgian handwriting reader, from scratch
#
# In the spirit of Karpathy's "Neural Networks: Zero to Hero" — every line explained,
# no magic, no hidden abstractions. By the end you will have trained a CRNN that reads
# handwritten Georgian text lines, and you'll understand *every* moving part.
#
# **The story so far**: we built a generator (github.com/LeonardoDaviti/GeoScribe) that
# renders unlimited synthetic Georgian handwriting — 13 hand-style fonts + drawn
# trajectories, deformations, and phone-photo degradations. Now we teach a network to
# read it. The end goal: digitize a real handwritten Georgian notebook from phone photos.
#
# **Hardware**: a free Colab T4 (16GB) is enough for everything in this notebook.
# Chapter 9 has the exact budget math.
#
# Lessons:
#   1. Look at the data (always, always look at the data first)
#   2. The charset — how text becomes numbers
#   3. Dataset & DataLoader — variable-width images, padding, collation
#   4. The CRNN — a model you can read in one sitting
#   5. CTC loss — the trick that makes OCR trainable without letter positions
#   6. The training loop — AMP, schedules, checkpoints
#   7. Evaluation — CER, error analysis, per-degradation breakdown
#   8. Read YOUR handwriting — inference on a photo
#   9. Experiments — the data-scaling laws you can measure yourself

# %% [markdown]
# ## Chapter 0 — Setup
#
# Get the dataset into Colab. On your machine, zip it first:
# `cd GeoScribe && zip -rq geoscribe_v01.zip v0.1` (~120MB for 100k lines),
# then either upload to Google Drive (fast to remount later) or upload directly here.

# %%
import os, json, math, random, time, zipfile
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device, '|', torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU only — expect ~20x slower')

# Option A: from Google Drive (uncomment):
# from google.colab import drive; drive.mount('/content/drive')
# DATA_ZIP = '/content/drive/MyDrive/geoscribe_v01.zip'
# Option B: direct upload (uncomment):
# from google.colab import files; files.upload()  # pick geoscribe_v01.zip
DATA_ZIP = 'geoscribe_v01.zip'
DATA_DIR = 'v0.1'

if not os.path.isdir(DATA_DIR):
    with zipfile.ZipFile(DATA_ZIP) as z:
        z.extractall('.')
print('images:', len(os.listdir(f'{DATA_DIR}/images')))

# %% [markdown]
# ## Chapter 1 — Look at the data
#
# Rule zero of ML: *inspect your data before you model it.* Each sample is a WebP image
# (64px tall, variable width) plus one JSON line of metadata. The `split` field is a
# hash of the TEXT — the same sentence can never appear in both train and val, so
# validation measures reading, not memorization (we learned this the hard way).

# %%
rows = [json.loads(l) for l in open(f'{DATA_DIR}/metadata.jsonl', encoding='utf-8')]
print('total samples:', len(rows))
print('example row:', json.dumps(rows[0], ensure_ascii=False)[:200])

from collections import Counter
print('splits:', Counter(r['split'] for r in rows))
print('degradations:', Counter(r['degrade'] for r in rows))
print('sources:', Counter(r['source'] for r in rows))

# display a few (in Colab this renders inline)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(6, 1, figsize=(14, 8))
for ax, r in zip(axes, random.sample(rows, 6)):
    ax.imshow(Image.open(f"{DATA_DIR}/{r['file_name']}").convert('L'), cmap='gray', aspect='auto')
    ax.set_title(f"[{r['degrade']}] {r['text']}", fontsize=9)
    ax.axis('off')
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Chapter 2 — The charset: text becomes numbers
#
# Neural nets eat integers, not letters. We map every character to a class index.
# Georgian is friendly here: 33 letters, no upper/lowercase, plus digits and
# punctuation — ~59 classes total (IAM English needs ~79).
#
# **Index 0 is reserved for the CTC blank** — a special "no letter here" symbol the
# loss function needs (Chapter 5). Real characters start at 1. This convention
# (blank=0) is what `torch.nn.CTCLoss` expects by default.

# %%
charset = sorted(set(ch for r in rows for ch in r['text']))
print(f'{len(charset)} characters:', ''.join(charset))

stoi = {ch: i + 1 for i, ch in enumerate(charset)}   # char -> int, 0 is CTC blank
itos = {i + 1: ch for i, ch in enumerate(charset)}   # int -> char

def encode(text): return [stoi[c] for c in text]
def decode_ids(ids): return ''.join(itos[i] for i in ids if i != 0)

print('encode("გამარჯობა") =', encode('გამარჯობა'))
assert decode_ids(encode('გამარჯობა')) == 'გამარჯობა'

# %% [markdown]
# ## Chapter 3 — Dataset & DataLoader
#
# Two non-obvious problems to solve:
#
# 1. **Variable widths.** "და" is ~100px wide, a full sentence ~1500px. GPU tensors are
#    rectangular, so a batch must be padded to its widest member. We pad with the
#    background value and remember each true width — the model must know where real
#    content ends (CTC needs it).
# 2. **Pixel scaling.** Raw pixels are 0..255. We map to [-1, 1] — centered inputs keep
#    early training stable (a plain rescale, nothing fancier needed here).
#
# The collate function is where batching actually happens — read it slowly.

# %%
class LineDataset(Dataset):
    def __init__(self, rows, data_dir, max_w=1600):
        self.rows, self.dir, self.max_w = rows, data_dir, max_w
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(f"{self.dir}/{r['file_name']}").convert('L')
        if img.height != 64:  # safety: normalize height, keep aspect
            img = img.resize((max(8, round(img.width * 64 / img.height)), 64))
        if img.width > self.max_w:
            img = img.resize((self.max_w, 64))
        x = torch.from_numpy(np.asarray(img, dtype=np.float32)) / 127.5 - 1.0  # [64, W] in [-1,1]
        y = torch.tensor(encode(r['text']), dtype=torch.long)
        return x, y, r

def collate(batch):
    xs, ys, rs = zip(*batch)
    W = max(x.shape[1] for x in xs)
    imgs = torch.full((len(xs), 1, 64, W), 1.0)          # pad with background (white=+1)
    widths = torch.tensor([x.shape[1] for x in xs])
    for i, x in enumerate(xs):
        imgs[i, 0, :, :x.shape[1]] = x
    targets = torch.cat(ys)                               # CTC wants targets concatenated flat
    target_lengths = torch.tensor([len(y) for y in ys])
    return imgs, widths, targets, target_lengths, rs

train_rows = [r for r in rows if r['split'] == 'train']
val_rows   = [r for r in rows if r['split'] == 'val']
print(f'train {len(train_rows)} / val {len(val_rows)}')

BATCH = 64
train_dl = DataLoader(LineDataset(train_rows, DATA_DIR), batch_size=BATCH, shuffle=True,
                      collate_fn=collate, num_workers=2, pin_memory=True, drop_last=True)
val_dl   = DataLoader(LineDataset(val_rows, DATA_DIR), batch_size=BATCH, shuffle=False,
                      collate_fn=collate, num_workers=2, pin_memory=True)

imgs, widths, targets, tlens, _ = next(iter(train_dl))
print('batch imgs:', tuple(imgs.shape), '| widths:', widths[:6].tolist(), '| targets flat:', tuple(targets.shape))

# %% [markdown]
# ## Chapter 4 — The CRNN, layer by layer
#
# The model never segments letters. Instead it converts the image into a **sequence of
# vertical slices** and classifies each slice. Three stages:
#
# 1. **CNN** — convolutions + pooling shrink 64px of height down to 1 while keeping
#    width (divided by 4). The output is a sequence: one feature vector per 4-pixel-wide
#    column of the original image. Imagine reading the line through a moving slit.
# 2. **BiLSTM** — each slice alone is ambiguous (a vertical stroke: is it ი or part of
#    ღ?). The bidirectional LSTM lets every position see its whole left and right
#    context. This is where letters-in-context happen.
# 3. **Head** — a linear layer maps each position to logits over (1 blank + N chars).
#
# Watch the printed shapes — that's the whole story of the architecture.

# %%
class CRNN(nn.Module):
    def __init__(self, n_classes, verbose=False):
        super().__init__()
        self.verbose = verbose
        def block(cin, cout, pool):  # conv -> BN -> ReLU -> pool
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(pool))
        self.cnn = nn.Sequential(
            block(1,   64, (2, 2)),   # 64x W    -> 32 x W/2
            block(64, 128, (2, 2)),   # 32x W/2  -> 16 x W/4
            block(128, 256, (2, 1)),  # 16x W/4  ->  8 x W/4   (stop shrinking width!)
            block(256, 256, (2, 1)),  #  8x W/4  ->  4 x W/4
            nn.Conv2d(256, 256, (4, 3), padding=(0, 1)), nn.ReLU(inplace=True),  # 4 -> 1
        )
        self.rnn = nn.LSTM(256, 256, num_layers=2, bidirectional=True, batch_first=False, dropout=0.1)
        self.head = nn.Linear(512, n_classes + 1)   # +1 for the CTC blank

    def forward(self, x):                       # x: [B, 1, 64, W]
        f = self.cnn(x)                         # [B, 256, 1, W/4]
        if self.verbose: print('after cnn:', tuple(f.shape))
        f = f.squeeze(2).permute(2, 0, 1)       # -> [T=W/4, B, 256]  (seq-first for LSTM/CTC)
        if self.verbose: print('as sequence:', tuple(f.shape))
        f, _ = self.rnn(f)                      # [T, B, 512]
        return self.head(f)                     # [T, B, n_classes+1]

model = CRNN(len(charset), verbose=True).to(device)
with torch.no_grad():
    logits = model(imgs[:2].to(device))
print('logits:', tuple(logits.shape), ' <- [time, batch, classes]')
print(f'parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
model.verbose = False
DOWNSAMPLE = 4   # the CNN divides width by this; CTC needs to know

# %% [markdown]
# ## Chapter 5 — CTC: the trick that makes this trainable
#
# The label says "გამარჯობა" (9 chars). The model produced, say, 200 predictions.
# Which prediction corresponds to which letter? **Nobody knows — and CTC's insight is
# that you don't have to decide.**
#
# CTC defines a *collapse rule*: merge consecutive repeats, then delete blanks (∅).
#   `გგგ∅∅ააა∅მმ...` -> `გამ...`  and  `გ∅გ` -> `გგ` (blank separates real doubles!)
#
# The loss = -log P(any alignment that collapses to the target). Summing over all
# valid alignments sounds exponential; a dynamic program does it in O(T·L). The model
# then *discovers* letter positions on its own, as a byproduct of maximizing this.
#
# Decoding (inference) is the same rule greedily: argmax per step -> collapse.

# %%
def greedy_decode(logits, widths=None):
    """logits [T,B,C] -> list of strings via argmax + CTC collapse."""
    T = logits.shape[0]
    best = logits.argmax(2).T.cpu().numpy()            # [B, T]
    out = []
    for b, seq in enumerate(best):
        t_end = T if widths is None else max(1, int(widths[b]) // DOWNSAMPLE)
        prev, chars = 0, []
        for t in range(t_end):
            k = seq[t]
            if k != 0 and k != prev: chars.append(itos[k])
            prev = k
        out.append(''.join(chars))
    return out

# toy demo of the collapse rule on a hand-built logit sequence:
toy = torch.full((7, 1, len(charset) + 1), -5.0)
for t, k in enumerate([stoi['გ'], stoi['გ'], 0, stoi['ა'], 0, stoi['მ'], stoi['მ']]):
    toy[t, 0, k] = 5.0
print("toy alignment გგ∅ა∅მმ collapses to:", greedy_decode(toy))  # -> 'გამ'

ctc = nn.CTCLoss(blank=0, zero_infinity=True)  # zero_infinity: don't explode when T < L

# %% [markdown]
# ## Chapter 6 — The training loop
#
# The eternal rhythm: forward, loss, backward, step. Plus three practical upgrades:
# **AMP** (fp16 on T4 tensor cores ≈ 2x speed), **OneCycle LR** (fast warmup, smooth
# decay — reliably good without babysitting), **gradient clipping** (LSTMs sometimes
# spike). We validate each epoch and checkpoint when val CER improves.
#
# CER = edit distance / length — implemented from scratch below, because you should
# see once in your life that "the metric" is 12 lines of dynamic programming.

# %%
def edit_distance(a, b):
    """Levenshtein: min insertions+deletions+substitutions to turn a into b."""
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            cur = min(dp[j] + 1,                       # delete
                      dp[j - 1] + 1,                   # insert
                      prev + (a[i - 1] != b[j - 1]))   # substitute (free if equal)
            prev, dp[j] = dp[j], cur
    return dp[-1]

@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    errs = chars = 0
    per_preset = {}
    for imgs, widths, targets, tlens, rs in dl:
        logits = model(imgs.to(device))
        preds = greedy_decode(logits, widths)
        for pred, r in zip(preds, rs):
            e = edit_distance(pred, r['text'])
            errs += e; chars += len(r['text'])
            p = per_preset.setdefault(r['degrade'], [0, 0])
            p[0] += e; p[1] += len(r['text'])
    model.train()
    return errs / max(1, chars), {k: v[0] / max(1, v[1]) for k, v in per_preset.items()}

EPOCHS = 15
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3,
        total_steps=EPOCHS * len(train_dl))
scaler = torch.amp.GradScaler()
best_cer = 1e9

for epoch in range(EPOCHS):
    t0, running = time.time(), 0.0
    for step, (imgs, widths, targets, tlens, _) in enumerate(train_dl):
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            logits = model(imgs)                                  # [T, B, C]
            logp = F.log_softmax(logits, dim=2)
            input_lengths = torch.clamp(widths // DOWNSAMPLE, min=1)
            loss = ctc(logp, targets, input_lengths, tlens)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt); scaler.update(); sched.step()
        running += loss.item()
        if step % 100 == 0:
            print(f'  e{epoch} s{step}/{len(train_dl)} loss {running/(step+1):.3f} lr {sched.get_last_lr()[0]:.2e}')
    cer, per_preset = evaluate(model, val_dl)
    print(f'epoch {epoch}: loss {running/len(train_dl):.3f} | val CER {cer*100:.2f}% | '
          f'{time.time()-t0:.0f}s | per-preset: ' +
          ' '.join(f'{k}:{v*100:.1f}%' for k, v in sorted(per_preset.items())))
    if cer < best_cer:
        best_cer = cer
        torch.save({'model': model.state_dict(), 'charset': charset}, 'geoscribe_crnn_best.pt')
        print(f'  ✓ new best, checkpoint saved ({best_cer*100:.2f}%)')
# tip: also copy the checkpoint to Drive so a Colab disconnect doesn't eat it:
# !cp geoscribe_crnn_best.pt /content/drive/MyDrive/

# %% [markdown]
# ## Chapter 7 — Error analysis: where does it fail?
#
# A single CER number hides everything interesting. Three lenses:
# 1. **Per-degradation CER** (printed each epoch above) — is `phone-hard` 5x worse than
#    `clean`? Then robustness, not reading, is the bottleneck.
# 2. **Worst samples** — look at them. Blurry? Rare letters? Long lines?
# 3. **Confusion pairs** — Georgian has known lookalikes (ღ/დ, ღ/ლ, უ/ყ, ვ/გ). Count
#    substitutions to see if the model struggles where humans do.

# %%
model.load_state_dict(torch.load('geoscribe_crnn_best.pt')['model'])
model.eval()
worst = []
with torch.no_grad():
    for imgs, widths, targets, tlens, rs in val_dl:
        preds = greedy_decode(model(imgs.to(device)), widths)
        for pred, r in zip(preds, rs):
            cer_i = edit_distance(pred, r['text']) / max(1, len(r['text']))
            worst.append((cer_i, r, pred))
worst.sort(key=lambda t: -t[0])
for cer_i, r, pred in worst[:8]:
    print(f'CER {cer_i*100:5.1f}% [{r["degrade"]:10s}] gt:   {r["text"]}')
    print(f'{"":25s} pred: {pred}')

# %% [markdown]
# ## Chapter 8 — Read YOUR handwriting
#
# The moment of truth. Photograph one *line* of your notebook (crop it to just the
# line, roughly horizontal), upload, and run. Expect worse than synthetic-val CER —
# quantifying that gap is the whole research program: every generator improvement
# (your drawn hand profile, real paper backgrounds, degradation tuning) exists to
# close it.

# %%
# from google.colab import files; up = files.upload(); path = next(iter(up))
path = 'my_line.jpg'  # <- or set a path directly
if os.path.exists(path):
    img = Image.open(path).convert('L')
    img = img.resize((max(8, round(img.width * 64 / img.height)), 64))
    x = torch.from_numpy(np.asarray(img, dtype=np.float32)) / 127.5 - 1.0
    with torch.no_grad():
        out = greedy_decode(model(x[None, None].to(device)))
    plt.figure(figsize=(14, 2)); plt.imshow(img, cmap='gray', aspect='auto'); plt.axis('off'); plt.show()
    print('model reads:', out[0])

# %% [markdown]
# ## Chapter 9 — Experiments & the T4 budget
#
# **T4 budget (16GB, ~free Colab)**: this CRNN (~6M params) trains at roughly
# 300-600 img/s with AMP -> a 95k-line epoch ≈ 3-6 min -> 15 epochs ≈ 1-1.5h. Fits a
# free session with headroom. Models up to ~50M params are comfortable; beyond that
# (TrOCR-base 334M) a T4 still *fits* it in fp16 but epochs stretch to hours —
# checkpoint to Drive and resume across sessions, or use a rented 4090.
#
# **Experiments to run** (each is one variable, one plot — this is the science):
#   A. Data scaling: train on 10k / 25k / 50k / 95k samples (subsample `train_rows`),
#      plot val CER vs data size. Log-linear? Where does it bend?
#   B. Degradation ablation: train on clean-only vs mix, evaluate both ways —
#      quantifies how much robustness costs/buys in clean accuracy.
#   C. Capacity: halve/double the LSTM width (256 -> 128/512). Params vs CER.
#   D. The real gap: as your real transcribed-lines eval set grows, track
#      synthetic-val CER vs real CER for every model above. That pair of curves is
#      the thesis figure.
#
# Change one thing at a time. Keep every run's config + result in a spreadsheet.
# That discipline is 80% of what separates research from vibes.
