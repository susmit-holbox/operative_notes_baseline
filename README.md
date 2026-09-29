# Operative notes for HeiCo-FOCUS videos (A_s189)

`generate_notes.py` runs **A_s189** (Qwen/Qwen3.5-9B + LoRA, merged) over the 30
colorectal-laparoscopy videos in the gated dataset `orena-dkfz/heico-focus-vqa` and
writes one free-text **operative note** per video.

Because A_s189 is a single-frame, ≤32-token VQA model with no cross-frame memory, the
script fakes "video → note" in two stages with the *same* model:

1. sample frames at a fixed interval and describe each one;
2. feed the accumulated timeline back as text and ask it to write the note.

## Prerequisites

- NVIDIA GPU with **~20 GB VRAM** (bf16 Qwen3.5-9B ≈ 18 GB; an A10/24 GB, L4/24 GB,
  or any A100/H100 works).
- `ffmpeg` on `PATH`.
- Python 3.10+ and:
  ```bash
  pip install torch transformers "peft==0.20.0" accelerate pillow huggingface_hub
  ```
  Pin `peft==0.20.0` if you re-merge from base + adapter (the adapter was built with
  peft 0.19.1/0.20.0). Newer `peft` adds a torchao version check and fails on machines
  where torchao is installed but `<0.16.0`.

## Model sourcing

Point `--model-path` at **one** of:

- **Route A — merged weights (what was evaluated).** The `resources/model/` directory
  inside the submitted container `frame-algorithm_remap_A_s189.tar.gz` (19.3 GB OCI
  image), or any merged `model/` directory already synced to S3.
- **Route B — re-merge base + adapter (lighter).** Pass the public base as
  `--model-path Qwen/Qwen3.5-9B` plus `--adapter-path
  downloads/orena-frame-weights/A_s189` (the 820 MB `adapter_model.safetensors` +
  `adapter_config.json`). The script does `PeftModel(...).merge_and_unload()`.
  Validate route B against route A on one known VQA row before trusting it.

## Run

Dataset access needs the HF token in the environment (`HF_TOKEN`).

Pilot — one local video, sparse sampling:

```bash
python generate_notes.py \
  --model-path /path/to/merged-A_s189 \
  --video "0000 - Heico - Prokto - 1.avi" \
  --sample-interval-sec 10 \
  --output-dir notes
```

Full sweep — stream all 30 videos from the hub, one at a time:

```bash
HF_TOKEN=hf_... python generate_notes.py \
  --model-path /path/to/merged-A_s189 \
  --download \
  --sample-interval-sec 5 \
  --output-dir notes
```

Useful flags: `--only <substring>` (one hub video), `--max-videos N`,
`--video-dir <dir>` (local directory of `.avi`/`.mp4`), `--batch-size N`,
`--adapter-path <dir>` (route B).

## Disk behaviour

Videos are processed **one at a time**. Frames are extracted to a temp dir, described,
then deleted; downloaded videos are deleted after their note is written. Peak disk is
one video (~3–9 GB) plus its frames (~1 GB), not the full ~170 GB corpus.

## Output

`notes/<video-stem>.md` per video, containing a `## Frame timeline` (timestamped
per-frame descriptions) and the final `## Operative note`.

## Caveats (read before trusting the output)

- A_s189's LoRA was fine-tuned for short VQA answers. Long prose generation is
  out-of-distribution for it; expect terse or repetitive notes on some videos. This is
  the best A_s189 alone can do — it is not a validated operative-report generator.
- Single-frame model: the "video memory" is simulated by text only. Temporal coherence
  depends entirely on the synthesis turn, which truncates the timeline to ~12k chars.
- Domain shift: A_s189 was trained on cholecystectomy + prostatectomy; HeiCo-FOCUS is
  colorectal. Descriptions may leak cholecystectomy-flavoured priors.
- **The notes are derivatives of a gated dataset.** HeiCo-FOCUS terms: no publication
  before the first challenge publication, and no redistribution of data/derivatives
  without the recipient accepting the same clause. Keep the notes local/private; do not
  push them to a public repo.

## Verification

1. Model sanity (route B only): one known VQA row must reproduce A_s189's answer.
2. Pilot on one video: frames extract, per-frame descriptions are non-empty and
   surgical, and the synthesised note is coherent prose.
3. Throughput: check s/frame on the pilot to pick `--sample-interval-sec` for the full
   sweep (longest video ≈ 4.5 h ⇒ ~3.2k frames at 5 s spacing).
4. Disk: confirm peak usage is one video + one frame set, not the whole corpus.
