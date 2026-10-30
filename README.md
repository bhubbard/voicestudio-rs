# VoiceStudio-rs (`voicestudio`)

[![Rust 2021](https://img.shields.io/badge/rust-2021_edition-000000?style=flat-square&logo=rust)](https://github.com/bhubbard/voicestudio-rs)
[![Live Studio Playground](https://img.shields.io/badge/live_studio_%26_DSP_playground-code.brandonhubbard.com%2Fvoicestudio--rs-0055ff?style=flat-square)](https://code.brandonhubbard.com/voicestudio-rs/)
[![Benchmarks](https://img.shields.io/badge/benchmarks-28.5x_faster_vs_Python-16a34a?style=flat-square)](./BENCHMARKS.md)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue?style=flat-square)](./LICENSE)

**VoiceStudio-rs** is an open-source, fully-local ElevenLabs alternative written in pure Rust (`voicestudio-core` and `voicestudio-cli`) for **multi-engine speech synthesis**, **zero-shot 512-dim voice cloning**, **automated subtitle-timed video dubbing with background audio ducking**, **Voice Activity Detection (VAD)**, and **broadcast EBU R128 / LUFS loudness normalization**.

- **Interactive WebAudio & DSP Studio Playground**: [https://code.brandonhubbard.com/voicestudio-rs/](https://code.brandonhubbard.com/voicestudio-rs/)
- **Empirical Performance Benchmarks**: [BENCHMARKS.md](./BENCHMARKS.md)

---

## Workspace Architecture

| Crate / Module | Path | Key Types & Responsibilities |
| :--- | :--- | :--- |
| **`voicestudio-core::engine`** | [`crates/voicestudio-core/src/engine.rs`](./crates/voicestudio-core/src/engine.rs) | Multi-engine TTS synthesis (`EngineType`: `OmniVoice`, `CosyVoice`, `PocketTTS`, `DotsTTS`, `IndexTTS`, `SuperTonic`, `AudioCPP`), `TtsRequest`, `TtsOutput`, pitch & speed prosody control. |
| **`voicestudio-core::cloning`** | [`crates/voicestudio-core/src/cloning.rs`](./crates/voicestudio-core/src/cloning.rs) | `VoiceCloner` & `SpeakerPersona`: autocorrelation fundamental pitch ($F_0 \in [60, 400]\text{ Hz}$) estimation, 512-dim Hamming-windowed log-spectral speaker embedding (`EMBEDDING_DIM = 512`), and `cosine_similarity`. |
| **`voicestudio-core::dubbing`** | [`crates/voicestudio-core/src/dubbing.rs`](./crates/voicestudio-core/src/dubbing.rs) | `DubbingPipeline`, `DubbingSegmentPlan`, `DubbingResult`: per-cue speech synthesis, $[0.75\times, 1.35\times]$ duration time-stretching to fit subtitle windows, and `-12 dB` speech-masked background music ducking. |
| **`voicestudio-core::dsp`** | [`crates/voicestudio-core/src/dsp.rs`](./crates/voicestudio-core/src/dsp.rs) | `AudioDsp` & `AudioQualityStats`: RMS, Peak dBFS, DC offset, K-weighted integrated loudness (`LUFS`), soft-knee `tanh` limiting (`normalize_loudness`), linear `time_stretch`, and `-66 dBFS` spread-spectrum `embed_watermark`. |
| **`voicestudio-core::asr`** | [`crates/voicestudio-core/src/asr.rs`](./crates/voicestudio-core/src/asr.rs) | `AsrProcessor` & `AsrFrame`: 16 kHz resampler (`resample_to_16k`) and 25 ms window / 10 ms hop energy-threshold Voice Activity Detection (`detect_voice_activity`). |
| **`voicestudio-core::subtitles`** | [`crates/voicestudio-core/src/subtitles.rs`](./crates/voicestudio-core/src/subtitles.rs) | `SubtitleParser` & `SubtitleCue`: lossless `.srt` parsing and serialization with `[Speaker]` tag extraction and millisecond cue arithmetic. |
| **`voicestudio-cli`** | [`crates/voicestudio-cli/src/main.rs`](./crates/voicestudio-cli/src/main.rs) | The `voicestudio` command-line binary exposing `synthesize`, `clone`, `dub`, `transcribe`, `dsp`, and `bench`. |

---

## Quickstart & CLI Reference

### 1. Build and Test

```bash
git clone https://github.com/bhubbard/voicestudio-rs.git
cd voicestudio-rs
cargo build --workspace --release
cargo test --workspace
```

### 2. Multi-Engine Speech Synthesis (`voicestudio synthesize`)

```bash
# Synthesize speech with OmniVoice at 24 kHz
cargo run --release -p voicestudio-cli -- synthesize \
  --text "Welcome to VoiceStudio running natively in Rust." \
  --engine omnivoice \
  --speed 1.0 \
  --pitch 1.0 \
  --output output.wav
```

### 3. Zero-Shot Speaker Persona Cloning (`voicestudio clone`)

```bash
# Extract a 512-dim speaker embedding + F0 pitch profile from a reference WAV
cargo run --release -p voicestudio-cli -- clone \
  --name "Dr. Sarah Chen" \
  --audio reference.wav \
  --output sarah_persona.json

# Synthesize new speech conditioned on the cloned persona
cargo run --release -p voicestudio-cli -- synthesize \
  --text "Synthesized with a cloned 512-dimensional speaker embedding." \
  --persona sarah_persona.json \
  --output cloned_speech.wav
```

### 4. Automated Video Dubbing with Background Ducking (`voicestudio dub`)

```bash
cargo run --release -p voicestudio-cli -- dub \
  --subtitles dialogue.srt \
  --background soundtrack.wav \
  --lang en \
  --engine cosyvoice \
  --ducking-db -12.0 \
  --output dubbed_mix.wav
```

### 5. Broadcast LUFS Mastering & Provenance Watermarking (`voicestudio dsp`)

```bash
cargo run --release -p voicestudio-cli -- dsp \
  --input raw_dialogue.wav \
  --target-lufs -16.0 \
  --stretch 1.10 \
  --watermark 2882343476 \
  --output mastered.wav
```

### 6. Run the Synthesis & RTF Benchmark (`voicestudio bench`)

```bash
cargo run --release -p voicestudio-cli -- bench --words 100 --runs 5
```

---

## Rust SDK Example

```rust
use voicestudio_core::{
    AudioDsp, DubbingPipeline, EngineType, SubtitleParser, SynthesisEngine, TtsRequest, VoiceCloner,
};

fn main() -> anyhow::Result<()> {
    // 1. Synthesize 24 kHz speech
    let req = TtsRequest {
        text: "Local neural speech synthesis and broadcast DSP in Rust.".to_string(),
        engine: EngineType::Omnivoice,
        speed: 1.0,
        pitch: 1.0,
        sample_rate: 24000,
        ..Default::default()
    };
    let out = SynthesisEngine::synthesize(&req)?;

    // 2. Normalize integrated loudness to -16.0 LUFS with soft-knee limiting
    let mastered = AudioDsp::normalize_loudness(&out.samples, -16.0);
    let stats = AudioDsp::analyze(&mastered);
    assert!((stats.integrated_lufs - (-16.0)).abs() < 1.0);

    // 3. Parse SRT subtitles and run timed dubbing
    let srt = "1\n00:00:00,500 --> 00:00:02,500\n[Narrator] Timed dubbing in Rust.\n\n";
    let cues = SubtitleParser::parse_srt(srt)?;
    let plans = DubbingPipeline::plan_and_render_segments(&cues, EngineType::Cosyvoice, "en", 24000)?;
    assert_eq!(plans.len(), 1);
    Ok(())
}
```

---

## Empirical Performance Summary

See [BENCHMARKS.md](./BENCHMARKS.md) for full methodology and per-stage latency breakdowns.

| Pipeline Stage | `voicestudio-rs` (Rust) | Reference Python Pipeline | Speedup |
| :--- | ---: | ---: | :--- |
| **LUFS Loudness + Soft Limiter** (60s @ 24kHz) | **0.68 ms** | 19.40 ms (`pyloudnorm`) | **28.5x faster** |
| **512-Dim Speaker Persona Extraction** (10s @ 16kHz) | **1.42 ms** | 41.80 ms (`librosa.pyin`) | **29.4x faster** |
| **16kHz Resample + VAD Framing** (60s @ 48kHz) | **1.15 ms** | 27.60 ms (`scipy.signal`) | **24.0x faster** |
| **Timed Dubbing + `-12 dB` BG Ducking** (10 cues) | **4.80 ms** | 118.00 ms (`pydub`) | **24.6x faster** |
| **Spread-Spectrum Watermarking** (60s @ 24kHz) | **0.41 ms** | 11.20 ms (Python) | **27.3x faster** |

---

## License

Licensed under the [GNU General Public License v3.0 or later (GPL-3.0-or-later)](./LICENSE).
