# Benchmarks

See the root [BENCHMARKS.md](../BENCHMARKS.md) for the complete empirical performance report comparing `voicestudio-rs` (`voicestudio-core` and `voicestudio-cli`) against the reference Python pipeline.

## Summary Table

| Pipeline Stage / Kernel | Rust Module (`voicestudio-core`) | `voicestudio-rs` (Rust) | Reference Python Pipeline | Speedup |
| :--- | :--- | ---: | ---: | :--- |
| **LUFS Loudness + Soft Limiter** | `dsp::AudioDsp::normalize_loudness` (60s @ 24kHz) | **0.68 ms** | 19.40 ms (`pyloudnorm` + `numpy`) | **28.5x faster** |
| **Speaker Persona Extraction** | `cloning::VoiceCloner::extract_persona` (10s @ 16kHz) | **1.42 ms** | 41.80 ms (`librosa.pyin` + filterbank) | **29.4x faster** |
| **ASR 16kHz Resample + VAD** | `asr::AsrProcessor::detect_voice_activity` (60s @ 48kHz) | **1.15 ms** | 27.60 ms (`scipy.signal` + framing) | **24.0x faster** |
| **Dubbing Stretch & Ducking Mix** | `dubbing::DubbingPipeline::mix_with_background` (10 cues) | **4.80 ms** | 118.00 ms (`pydub` / segment stitch) | **24.6x faster** |
| **Provenance Watermarking** | `dsp::AudioDsp::embed_watermark` (60s @ 24kHz) | **0.41 ms** | 11.20 ms (Python spread-spectrum) | **27.3x faster** |
