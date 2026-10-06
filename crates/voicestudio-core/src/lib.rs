//! voicestudio-core: High-performance core engine for VoiceStudio
//!
//! Provides multi-engine speech synthesis (CosyVoice, OmniVoice, PocketTTS, DotsTTS, IndexTTS, SuperTonic, AudioCPP),
//! zero-shot voice cloning and speaker persona embeddings, video dubbing duration planning & mixing,
//! audio DSP loudness normalization, and ASR speech processing.

pub mod asr;
pub mod cloning;
pub mod config;
pub mod dsp;
pub mod dubbing;
pub mod engine;
pub mod subtitles;

pub use asr::{AsrFrame, AsrProcessor, ASR_SAMPLE_RATE};
pub use cloning::{SpeakerPersona, VoiceCloner, EMBEDDING_DIM};
pub use config::VoiceStudioConfig;
pub use dsp::{AudioDsp, AudioQualityStats};
pub use dubbing::{DubbingPipeline, DubbingResult, DubbingSegmentPlan};
pub use engine::{EngineType, SynthesisEngine, TtsOutput, TtsRequest};
pub use subtitles::{SubtitleCue, SubtitleParser};
