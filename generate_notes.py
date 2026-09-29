#!/usr/bin/env python3
"""Generate per-video operative notes for HeiCo-FOCUS videos using A_s189.

A_s189 is Qwen/Qwen3.5-9B + LoRA (r=64, alpha=128, merged into the base) — a
**single-frame, greedy, short-answer VQA** model. Its shipped inference runs with
``max_new_tokens=32`` (see ``submission/inference_A_s189.py``). It has no
cross-frame memory, so the only honest way to get a "video -> operative note" out
of it is two-stage:

  1. sample frames at a fixed interval and ask the model to describe each one;
  2. hand the accumulated timeline back to the same model as text and ask it to
     write a full operative note.

The model is loaded once. Videos are processed one at a time and their frames (and
the source video, when downloaded) are deleted as we go, so peak disk is bounded by
one video plus one video's worth of JPEGs (~10 GB for the longest clip), never the
full ~170 GB corpus.

Two settings are copied verbatim from ``inference_A_s189.py`` because they are
load-bearing and must not be "improved":

  * processor ``size={"longest_edge": 1048576, "shortest_edge": 65536}``
  * ``eos_token_id=[248044, 248046]`` — Qwen3.5-9B ships no generation_config.json
    and the chat template closes a turn with ``<|im_end|>`` (248046) while the
    config's EOS is ``<|endoftext|>`` (248044). Without both, generation never stops.

Not copied: ``finalise`` (the 300-char cap) and ``remap_answer`` (VQA priors) — they
would truncate and corrupt prose.

Usage (see README.md for the full cloud run recipe):

  # local video file, pilot run
  python generate_notes.py --model-path /path/to/merged-A_s189 \
      --video "0000 - Heico - Prokto - 1.avi" --sample-interval-sec 10

  # whole gated dataset, streamed from the hub (token in HF_TOKEN)
  HF_TOKEN=... python generate_notes.py --model-path /path/to/merged-A_s189 --download

  # re-merge from base + adapter instead of a pre-merged directory
  python generate_notes.py --model-path Qwen/Qwen3.5-9B \
      --adapter-path downloads/orena-frame-weights/A_s189 --video ...
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

log = logging.getLogger("operative-notes")

# ── Model / processor settings (copied from inference_A_s189.py) ──────────────
MAX_PIXELS = 1048576          # run_args.json: "max_pixels": 1048576
SHORTEST_EDGE = 65536         # processor kwarg
MAX_LENGTH = 4096             # run_args.json: "max_len": 4096

# ── Generation budgets (the one thing we change vs. the VQA submission) ──────
FRAME_DESC_MAX_NEW_TOKENS = 128    # per-frame caption, still short
NOTE_MAX_NEW_TOKENS = 2048         # final operative note

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_FRAME = "You are a surgical assistant observing a laparoscopic procedure."
FRAME_QUESTION = (
    "Describe what is happening in this surgical frame in one or two sentences."
)
SYSTEM_NOTE = "You are a surgical resident writing an operative note."
NOTE_QUESTION_TMPL = (
    "Write a complete operative note for this procedure, based on the timeline of "
    "frame descriptions below. Include the sections Indication, Procedure, Findings "
    "and Closure, written in full prose sentences like a real operative report.\n\n"
    "Timeline:\n{timeline}"
)

# Synthesis turns are text-only, so the accumulated timeline must fit the model's
# context window. ~12k chars is a safe bound under MAX_LENGTH=4096 tokens.
MAX_TIMELINE_CHARS = 12000

BATCH_SIZE_DEFAULT = 8
HF_REPO = "orena-dkfz/heico-focus-vqa"


def _fmt_ts(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _eos_ids(processor) -> list[int]:
    """``[248044, 248046]`` — identical to focus_common.eos_ids / inference_A_s189.py."""
    tok = processor.tokenizer
    ids = []
    for name in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(name)
        if tid is None or tid == tok.unk_token_id:
            raise RuntimeError(f"tokenizer does not know the EOS token {name!r}")
        ids.append(tid)
    return sorted(set(ids))


def _build_prompt(processor, question: str, system: str, with_image: bool) -> str:
    """Chat-template prompt. ``with_image=False`` renders a text-only turn."""
    q = str(question).replace("<image>", "").strip()
    user = ([{"type": "image"}, {"type": "text", "text": q}]
            if with_image else [{"type": "text", "text": q}])
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": user},
    ]
    return processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def _generate(model, processor, eos_ids, prompts, images, max_new_tokens):
    """Greedy generation. ``images`` is a list of PIL images, or None for text-only."""
    if images is not None:
        batch = processor(text=prompts, images=images, return_tensors="pt",
                          padding=True).to(model.device)
    else:
        batch = processor(text=prompts, return_tensors="pt", padding=True).to(model.device)
    n_in = batch["input_ids"].shape[1]
    if n_in > MAX_LENGTH:
        log.warning("batch input length %d exceeds max_length %d", n_in, MAX_LENGTH)
    with torch.inference_mode():
        gen = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=eos_ids,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    return processor.tokenizer.batch_decode(gen[:, n_in:], skip_special_tokens=True)


def load_model(model_path: str, adapter_path: str | None, device: str):
    """Load A_s189.

    * ``adapter_path`` set  -> treat ``model_path`` as the Qwen3.5-9B base and
      re-merge the LoRA on top (route B; lighter on a fresh box).
    * ``adapter_path`` None -> ``model_path`` is already the merged model (route A).
    """
    processor = AutoProcessor.from_pretrained(
        model_path,
        padding_side="left",
        size={"longest_edge": MAX_PIXELS, "shortest_edge": SHORTEST_EDGE},
    )
    dtype = torch.bfloat16
    if adapter_path:
        from peft import PeftModel
        base = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=dtype, device_map=device, attn_implementation="sdpa"
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=dtype, device_map=device, attn_implementation="sdpa"
        )
    model.eval()
    model.config.use_cache = True
    return model, processor


def extract_frames(video_path: Path, interval_sec: float, out_dir: Path) -> list[Path]:
    """Extract evenly spaced JPEGs with ffmpeg. Returns sorted frame paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%06d.jpg"
    fps = 1.0 / interval_sec
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={fps:.6f}",
        "-q:v", "3",
        str(pattern),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()[:500]}")
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("ffmpeg produced no frames")
    return frames


def _describe_frames(model, processor, eos_ids, frames, batch_size):
    """Describe every frame. Returns a list of (frame_path, description)."""
    prompt = _build_prompt(processor, FRAME_QUESTION, SYSTEM_FRAME, with_image=True)
    results = []
    for start in range(0, len(frames), batch_size):
        chunk = frames[start:start + batch_size]
        images, ok = [], []
        for fp in chunk:
            try:
                images.append(Image.open(fp).convert("RGB"))
                ok.append(fp)
            except Exception:
                log.exception("could not load frame %s; skipping", fp.name)
        if not images:
            continue
        texts = _generate(model, processor, eos_ids, [prompt] * len(images),
                          images, FRAME_DESC_MAX_NEW_TOKENS)
        results.extend(zip(ok, texts))
        log.info("    described %d/%d frames", start + len(ok), len(frames))
    return results


def _compress_timeline(lines: list[str], max_chars: int = MAX_TIMELINE_CHARS) -> list[str]:
    """Uniformly subsample timeline lines to fit ``max_chars``, keeping first+last."""
    if sum(len(l) + 1 for l in lines) <= max_chars:
        return lines
    lo, hi, stride = 1, len(lines), 1
    while lo <= hi:
        mid = (lo + hi) // 2
        chosen = lines[::mid]
        if chosen[-1] != lines[-1]:
            chosen = chosen + [lines[-1]]
        if sum(len(l) + 1 for l in chosen) <= max_chars:
            stride, lo = mid, mid + 1
        else:
            hi = mid - 1
    out = lines[::stride]
    if out[-1] != lines[-1]:
        out.append(lines[-1])
    log.info("    timeline subsampled %d -> %d lines to fit context", len(lines), len(out))
    return out


def _synthesize_note(model, processor, eos_ids, timeline: str, anchor_image=None):
    """Write the operative note from the timeline. Text-only, with a last-frame
    anchor fallback if the processor rejects an image-less turn."""
    question = NOTE_QUESTION_TMPL.format(timeline=timeline)
    prompt = _build_prompt(processor, question, SYSTEM_NOTE, with_image=False)
    try:
        return _generate(model, processor, eos_ids, [prompt], None, NOTE_MAX_NEW_TOKENS)[0]
    except Exception:
        if anchor_image is None:
            raise
        log.warning("text-only synthesis failed; retrying anchored on the last frame")
        prompt = _build_prompt(processor, question, SYSTEM_NOTE, with_image=True)
        return _generate(model, processor, eos_ids, [prompt], [anchor_image],
                         NOTE_MAX_NEW_TOKENS)[0]


def process_video(model, processor, eos_ids, video_path: Path, interval_sec: float,
                  batch_size: int) -> dict:
    """Full per-video pipeline. Returns a dict with 'video' and 'note'/'error'."""
    name = video_path.name
    log.info("== %s ==", name)
    tmp = Path(tempfile.mkdtemp(prefix="frames_"))
    try:
        frames = extract_frames(video_path, interval_sec, tmp)
        log.info("  extracted %d frames (1/%ss)", len(frames), interval_sec)
    except Exception as exc:
        log.exception("frame extraction failed for %s", name)
        shutil.rmtree(tmp, ignore_errors=True)
        return {"video": name, "error": str(exc)}

    anchor = None
    try:
        anchor = Image.open(frames[-1]).convert("RGB")
        descs = _describe_frames(model, processor, eos_ids, frames, batch_size)
    except Exception as exc:
        log.exception("frame description failed for %s", name)
        shutil.rmtree(tmp, ignore_errors=True)
        return {"video": name, "error": str(exc)}

    lines = [f"[{_fmt_ts(i * interval_sec)}] {t.strip()}"
             for i, (_, t) in enumerate(descs) if t.strip()]
    timeline = "\n".join(_compress_timeline(lines))
    shutil.rmtree(tmp, ignore_errors=True)

    if not timeline.strip():
        log.warning("no usable frame descriptions for %s", name)
        return {"video": name, "timeline": "", "note": "(no frame descriptions produced)"}

    log.info("  timeline: %d chars over %d lines", len(timeline), timeline.count("\n") + 1)
    try:
        note = _synthesize_note(model, processor, eos_ids, timeline, anchor)
    except Exception as exc:
        log.exception("note synthesis failed for %s", name)
        return {"video": name, "timeline": timeline, "error": str(exc)}

    return {"video": name, "timeline": timeline, "note": note.strip()}


def _list_dataset_videos(token: str) -> list[str]:
    from huggingface_hub import HfApi
    files = HfApi(token=token).list_repo_files(HF_REPO, repo_type="dataset")
    return sorted(f for f in files if f.startswith("videos/") and f.lower().endswith(".avi"))


def _download_video(token: str, filename: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        HF_REPO, filename=filename, repo_type="dataset", token=token,
        local_dir=str(dest_dir), cache_dir=str(dest_dir / ".cache"),
    ))


def write_note(out_dir: Path, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result["video"]).stem
    out_path = out_dir / f"{stem}.md"
    with open(out_path, "w") as f:
        f.write(f"# Operative note — {result['video']}\n\n")
        if result.get("error"):
            f.write(f"**ERROR:** {result['error']}\n\n")
        if result.get("timeline"):
            f.write("## Frame timeline\n\n```\n")
            f.write(result["timeline"])
            f.write("\n```\n\n")
        if result.get("note"):
            f.write("## Operative note\n\n")
            f.write(result["note"])
            f.write("\n")
    log.info("wrote %s", out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default=os.environ.get("A_S189_MODEL"),
                    help="merged A_s189 dir, or Qwen3.5-9B base when --adapter-path is set")
    ap.add_argument("--adapter-path", default=None,
                    help="A_s189 LoRA dir (base+adapter -> merge), for route B")
    ap.add_argument("--video", default=None, help="single local video file")
    ap.add_argument("--video-dir", default=None, help="directory of local .avi/.mp4/.mov")
    ap.add_argument("--download", action="store_true",
                    help="stream videos from the gated HF dataset")
    ap.add_argument("--only", default=None,
                    help="with --download: process only the video whose filename contains this")
    ap.add_argument("--max-videos", type=int, default=None, help="stop after N videos")
    ap.add_argument("--sample-interval-sec", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument("--output-dir", default="notes")
    args = ap.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    token = os.environ.get("HF_TOKEN")
    if args.download and not token:
        log.error("--download needs HF_TOKEN in the environment")
        return 2
    if not args.model_path:
        log.error("--model-path is required (or set A_S189_MODEL)")
        return 2
    if args.sample_interval_sec <= 0:
        log.error("--sample-interval-sec must be > 0")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    model, processor = load_model(args.model_path, args.adapter_path,
                                  "cuda:0" if device.type == "cuda" else "cpu")
    eos_ids = _eos_ids(processor)
    log.info("eos_token_id: %s", eos_ids)

    # ── Build the worklist: (display name, local path, delete_after) ──────────
    jobs: list[tuple[str, Path, bool]] = []
    if args.download:
        vids = _list_dataset_videos(token)
        if args.only:
            vids = [v for v in vids if args.only in v]
        if args.max_videos:
            vids = vids[:args.max_videos]
        jobs = [(Path(fn).name, Path(fn), True) for fn in vids]
    elif args.video:
        jobs = [(Path(args.video).name, Path(args.video), False)]
    elif args.video_dir:
        d = Path(args.video_dir)
        files = sorted(p for p in d.iterdir()
                       if p.suffix.lower() in (".avi", ".mp4", ".mov"))
        if args.max_videos:
            files = files[:args.max_videos]
        jobs = [(p.name, p, False) for p in files]
    else:
        log.error("choose one of --download / --video / --video-dir")
        return 2

    log.info("processing %d video(s)", len(jobs))
    out_dir = Path(args.output_dir)
    for name, _remote, delete_after in jobs:
        local = _remote
        dl_dir = None
        if delete_after:
            dl_dir = Path(tempfile.mkdtemp(prefix="video_"))
            try:
                local = _download_video(token, f"videos/{name}", dl_dir)
            except Exception as exc:
                log.exception("download failed for %s", name)
                shutil.rmtree(dl_dir, ignore_errors=True)
                write_note(out_dir, {"video": name, "error": str(exc)})
                continue
        try:
            result = process_video(model, processor, eos_ids, local,
                                   args.sample_interval_sec, args.batch_size)
            write_note(out_dir, result)
        finally:
            if dl_dir is not None:
                shutil.rmtree(dl_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
