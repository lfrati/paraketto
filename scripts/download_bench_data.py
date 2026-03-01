#!/usr/bin/env python3
"""Download and prepare benchmark audio data.

Downloads LibriSpeech test-clean and VoxPopuli en_accented, prepares:
  1. data/librispeech/ — 100 short utterances across 40 speakers (clean, 2-27s)
  2. data/long/        — 50 chapter-level files truncated to ≤120s (55-120s)
  3. data/difficult/   — 50 accented utterances from VoxPopuli (15 EU accents, mild noise)

All output is 16kHz mono WAV. Requires ffmpeg.

Usage:
    uv run python scripts/download_bench_data.py
"""

import json
import os
import struct
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

LIBRISPEECH_CLEAN_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
VOXPOPULI_PARQUET = "https://huggingface.co/datasets/facebook/voxpopuli/resolve/main/en_accented/test-00000-of-00002.parquet"

LIBRISPEECH_DIR = DATA_DIR / "librispeech"
LONG_DIR = DATA_DIR / "long"
DIFFICULT_DIR = DATA_DIR / "difficult"

TARGET_SHORT = 100
TARGET_LONG = 50
TARGET_DIFFICULT = 50
MAX_DURATION = 120  # seconds — matches CUDA backend limit
TARGET_SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, desc: str = "") -> None:
    """Download a file with progress reporting."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc or url}...")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req)
    total = int(resp.headers.get("Content-Length", 0))
    downloaded = 0
    chunk_size = 1 << 20  # 1MB

    with open(dest, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 // total
                mb = downloaded / (1 << 20)
                total_mb = total / (1 << 20)
                print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct}%)", end="", flush=True)
    print()


def decode_to_pcm16(input_path: str) -> bytes:
    """Decode any audio file to raw 16kHz mono PCM16 via ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_path,
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1",
            "pipe:1",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {input_path}: {result.stderr.decode()}")
    return result.stdout


def write_wav(path: Path, pcm_data: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    """Write a 16-bit mono WAV file from raw PCM data."""
    byte_rate = sample_rate * 2
    data_size = len(pcm_data)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm_data)


def pcm_duration(pcm_data: bytes) -> float:
    """Duration in seconds for mono 16-bit PCM at TARGET_SAMPLE_RATE."""
    return len(pcm_data) / (2 * TARGET_SAMPLE_RATE)


# ---------------------------------------------------------------------------
# LibriSpeech helpers
# ---------------------------------------------------------------------------

def parse_librispeech(base_dir: Path) -> dict:
    """Parse a LibriSpeech split into speaker -> chapter -> [(id, flac_path, transcript)]."""
    speaker_chapters = defaultdict(lambda: defaultdict(list))
    for speaker_dir in sorted(base_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker = speaker_dir.name
        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue
            chapter = chapter_dir.name
            trans_file = chapter_dir / f"{speaker}-{chapter}.trans.txt"
            if not trans_file.exists():
                continue
            transcripts = {}
            for line in trans_file.read_text().strip().split("\n"):
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    transcripts[parts[0]] = parts[1]
            for flac in sorted(chapter_dir.glob("*.flac")):
                utt_id = flac.stem
                if utt_id in transcripts:
                    speaker_chapters[speaker][chapter].append(
                        (utt_id, str(flac), transcripts[utt_id])
                    )
    return speaker_chapters


def select_round_robin(speaker_chapters: dict, target: int) -> list:
    """Select utterances spread evenly across speakers via round-robin."""
    all_utts_by_speaker = {}
    for speaker in sorted(speaker_chapters.keys()):
        utts = []
        for chapter in sorted(speaker_chapters[speaker].keys()):
            utts.extend(speaker_chapters[speaker][chapter])
        all_utts_by_speaker[speaker] = utts

    selected = []
    speaker_list = sorted(all_utts_by_speaker.keys())
    speaker_idx = {s: 0 for s in speaker_list}

    while len(selected) < target:
        added_any = False
        for speaker in speaker_list:
            if len(selected) >= target:
                break
            idx = speaker_idx[speaker]
            utts = all_utts_by_speaker[speaker]
            if idx < len(utts):
                selected.append((speaker, utts[idx]))
                speaker_idx[speaker] = idx + 1
                added_any = True
        if not added_any:
            break
    return selected


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_short(speaker_chapters: dict):
    """100 short utterances from test-clean, spread across 40 speakers."""
    print(f"\n--- Preparing {TARGET_SHORT} short LibriSpeech utterances ---")
    LIBRISPEECH_DIR.mkdir(parents=True, exist_ok=True)
    selected = select_round_robin(speaker_chapters, TARGET_SHORT)
    manifest = []
    speaker_counts = defaultdict(int)

    for i, (speaker, (utt_id, flac_path, transcript)) in enumerate(selected):
        wav_name = f"{utt_id}.wav"
        wav_path = LIBRISPEECH_DIR / wav_name
        if not wav_path.exists():
            pcm = decode_to_pcm16(flac_path)
            write_wav(wav_path, pcm)
            dur = pcm_duration(pcm)
        else:
            dur = (wav_path.stat().st_size - 44) / (2 * TARGET_SAMPLE_RATE)

        manifest.append({
            "id": utt_id,
            "audio_path": wav_name,
            "reference": transcript,
            "duration_s": round(dur, 2),
        })
        speaker_counts[speaker] += 1
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{TARGET_SHORT}")

    (LIBRISPEECH_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(e["duration_s"] for e in manifest)
    durs = [e["duration_s"] for e in manifest]
    print(f"  {len(manifest)} utterances, {len(speaker_counts)} speakers, "
          f"{total/60:.1f}min, {min(durs):.1f}s-{max(durs):.1f}s")


def prepare_long(speaker_chapters: dict):
    """50 chapter-level files truncated at utterance boundaries to ≤120s."""
    print(f"\n--- Preparing {TARGET_LONG} long chapter files (≤{MAX_DURATION}s) ---")
    LONG_DIR.mkdir(parents=True, exist_ok=True)

    candidates = []
    for speaker in sorted(speaker_chapters.keys()):
        for chapter in sorted(speaker_chapters[speaker].keys()):
            utts = speaker_chapters[speaker][chapter]
            pcm_parts = []
            transcripts = []
            total_dur = 0.0
            for utt_id, flac_path, transcript in utts:
                pcm = decode_to_pcm16(flac_path)
                dur = pcm_duration(pcm)
                if total_dur + dur > MAX_DURATION and total_dur > 0:
                    break
                pcm_parts.append(pcm)
                transcripts.append(transcript)
                total_dur += dur
            if total_dur < 30.0:
                continue
            candidates.append({
                "id": f"{speaker}-{chapter}",
                "pcm": b"".join(pcm_parts),
                "reference": " ".join(transcripts),
                "duration": total_dur,
            })

    candidates.sort(key=lambda c: c["duration"])
    print(f"  {len(candidates)} chapter candidates")

    if len(candidates) <= TARGET_LONG:
        selected = candidates
    else:
        step = len(candidates) / TARGET_LONG
        selected = [candidates[int(i * step)] for i in range(TARGET_LONG)]

    manifest = []
    for c in selected:
        wav_name = f"{c['id']}.wav"
        write_wav(LONG_DIR / wav_name, c["pcm"])
        manifest.append({
            "id": c["id"],
            "audio_path": wav_name,
            "reference": c["reference"],
            "duration_s": round(c["duration"], 2),
        })

    manifest.sort(key=lambda e: e["duration_s"])
    (LONG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(e["duration_s"] for e in manifest)
    durs = [e["duration_s"] for e in manifest]
    speakers = len(set(e["id"].split("-")[0] for e in manifest))
    print(f"  {len(manifest)} files, {speakers} speakers, "
          f"{total/60:.1f}min, {min(durs):.1f}s-{max(durs):.1f}s")


def prepare_difficult():
    """50 accented utterances from VoxPopuli en_accented (European Parliament).

    Downloads a HuggingFace parquet file, reads metadata to select 50 samples
    spread across 15 accents with unique speakers, then fetches only the
    needed audio rows. No auth required (CC0 license).
    """
    print(f"\n--- Preparing {TARGET_DIFFICULT} accented utterances (VoxPopuli) ---")

    try:
        import fsspec
        import pyarrow.parquet as pq
    except ImportError:
        print("  ERROR: pip install fsspec pyarrow  (needed for VoxPopuli download)")
        sys.exit(1)

    DIFFICULT_DIR.mkdir(parents=True, exist_ok=True)
    httpfs = fsspec.filesystem("https")

    # 1. Read metadata only (no audio bytes) to plan selection
    print("  Reading parquet metadata...")
    f = httpfs.open(VOXPOPULI_PARQUET)
    pf = pq.ParquetFile(f)
    meta_cols = ["audio_id", "raw_text", "normalized_text", "speaker_id", "accent"]
    all_rows = []
    for rg_idx in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg_idx, columns=meta_cols)
        for j in range(tbl.num_rows):
            row = {c: tbl.column(c).chunk(0)[j].as_py() for c in meta_cols}
            row["_rg"] = rg_idx
            row["_row"] = j
            all_rows.append(row)
    f.close()

    # 2. Round-robin across accents, picking unique speakers
    by_accent = defaultdict(list)
    for r in all_rows:
        by_accent[r["accent"]].append(r)
    accents = sorted(by_accent.keys())

    selected = []
    seen_speakers = set()
    accent_idx = {a: 0 for a in accents}
    while len(selected) < TARGET_DIFFICULT:
        added = False
        for accent in accents:
            if len(selected) >= TARGET_DIFFICULT:
                break
            for idx in range(accent_idx[accent], len(by_accent[accent])):
                r = by_accent[accent][idx]
                if r["speaker_id"] not in seen_speakers:
                    selected.append(r)
                    seen_speakers.add(r["speaker_id"])
                    accent_idx[accent] = idx + 1
                    added = True
                    break
        if not added:
            break

    by_rg = defaultdict(list)
    for i, s in enumerate(selected):
        by_rg[s["_rg"]].append((i, s))

    print(f"  Selected {len(selected)} samples, {len(set(s['accent'] for s in selected))} accents, "
          f"{len(seen_speakers)} speakers")

    # 3. Fetch audio row groups and convert to 16kHz WAV
    manifest = [None] * len(selected)
    for rg_idx, items in sorted(by_rg.items()):
        print(f"  Fetching row group {rg_idx} ({len(items)} samples)...")
        f2 = httpfs.open(VOXPOPULI_PARQUET)
        pf2 = pq.ParquetFile(f2)
        tbl = pf2.read_row_group(rg_idx)
        audio_chunk = tbl.column("audio").chunk(0)

        for sel_idx, s in items:
            audio_bytes = audio_chunk[s["_row"]].as_py()["bytes"]
            audio_id = s["audio_id"]

            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                pcm = decode_to_pcm16(tmp_path)
            except RuntimeError:
                print(f"    SKIP {audio_id}: decode error")
                continue
            finally:
                os.unlink(tmp_path)

            dur = pcm_duration(pcm)
            if dur > MAX_DURATION:
                pcm = pcm[:int(MAX_DURATION * 2 * TARGET_SAMPLE_RATE)]
                dur = MAX_DURATION

            wav_name = f"{audio_id}.wav"
            write_wav(DIFFICULT_DIR / wav_name, pcm)

            manifest[sel_idx] = {
                "id": audio_id,
                "audio_path": wav_name,
                "reference": s["normalized_text"] or s["raw_text"] or "",
                "duration_s": round(dur, 2),
                "accent": s["accent"],
                "speaker_id": s["speaker_id"],
            }
        f2.close()

    manifest = [m for m in manifest if m is not None]
    (DIFFICULT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    durs = [m["duration_s"] for m in manifest]
    accent_counts = Counter(m["accent"] for m in manifest)
    speakers = len(set(m["speaker_id"] for m in manifest))
    print(f"  {len(manifest)} files, {speakers} speakers, {len(accent_counts)} accents")
    print(f"  {sum(durs)/60:.1f}min total, {min(durs):.1f}s-{max(durs):.1f}s")
    print(f"  Accents: {dict(sorted(accent_counts.items()))}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Downloading benchmark audio data ===\n")

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: ffmpeg not found. Install with: sudo apt-get install -y ffmpeg")
        sys.exit(1)

    # Download and extract LibriSpeech test-clean
    tar_clean = DATA_DIR / "test-clean.tar.gz"
    download_file(LIBRISPEECH_CLEAN_URL, tar_clean, "LibriSpeech test-clean (346MB)")

    extract_dir = DATA_DIR / "_librispeech_raw"
    test_clean_dir = extract_dir / "LibriSpeech" / "test-clean"
    if not test_clean_dir.exists():
        print("  Extracting test-clean...")
        with tarfile.open(tar_clean) as tf:
            tf.extractall(extract_dir)
        print("  Extracted.")

    clean_chapters = parse_librispeech(test_clean_dir)
    print(f"  Found {len(clean_chapters)} speakers in test-clean")

    # 1. Short clean utterances
    prepare_short(clean_chapters)

    # 2. Long chapter-level audio (≤120s)
    prepare_long(clean_chapters)

    # Clean up LibriSpeech extraction
    print("\n  Cleaning up extracted LibriSpeech files...")
    subprocess.run(["rm", "-rf", str(extract_dir)])

    # 3. Accented utterances from VoxPopuli
    prepare_difficult()

    # Summary
    print("\n=== Done ===\n")
    print("Dataset summary:")
    for name, path in [("librispeech", LIBRISPEECH_DIR), ("long", LONG_DIR), ("difficult", DIFFICULT_DIR)]:
        mf = path / "manifest.json"
        if not mf.exists():
            continue
        entries = json.loads(mf.read_text())
        total = sum(e["duration_s"] for e in entries)
        durs = [e["duration_s"] for e in entries]
        speakers = len(set(e.get("speaker_id", e["id"].split("-")[0]) for e in entries))
        extra = ""
        if name == "difficult":
            extra = f", {len(set(e.get('accent','') for e in entries))} accents"
        print(f"  {name}: {len(entries)} files, {speakers} speakers, "
              f"{total/60:.1f}min total, {min(durs):.1f}s-{max(durs):.1f}s{extra}")


if __name__ == "__main__":
    main()
