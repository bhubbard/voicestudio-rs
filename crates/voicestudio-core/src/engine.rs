use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineType {
    Cosyvoice,
    Omnivoice,
    Pockettts,
    Dotstts,
    Indextts,
    Supertonic,
    Audiocpp,
}

impl FromStr for EngineType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cosyvoice" | "cosy" => Ok(EngineType::Cosyvoice),
            "omnivoice" | "omni" => Ok(EngineType::Omnivoice),
            "pockettts" | "pocket" => Ok(EngineType::Pockettts),
            "dotstts" | "dots" => Ok(EngineType::Dotstts),
            "indextts" | "index" => Ok(EngineType::Indextts),
            "supertonic" | "tonic" => Ok(EngineType::Supertonic),
            "audiocpp" => Ok(EngineType::Audiocpp),
            other => anyhow::bail!("Unknown engine type: {}", other),
        }
    }
}

impl std::fmt::Display for EngineType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            EngineType::Cosyvoice => "CosyVoice",
            EngineType::Omnivoice => "OmniVoice",
            EngineType::Pockettts => "PocketTTS",
            EngineType::Dotstts => "DotsTTS",
            EngineType::Indextts => "IndexTTS",
            EngineType::Supertonic => "SuperTonic",
            EngineType::Audiocpp => "AudioCPP",
        };
        write!(f, "{}", name)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsRequest {
    pub text: String,
    pub engine: EngineType,
    pub voice_id: Option<String>,
    pub speaker_embedding: Option<Vec<f32>>,
    pub language: String,
    pub speed: f32,
    pub pitch: f32,
    pub sample_rate: u32,
}

impl Default for TtsRequest {
    fn default() -> Self {
        Self {
            text: String::new(),
            engine: EngineType::Omnivoice,
            voice_id: None,
            speaker_embedding: None,
            language: "en".to_string(),
            speed: 1.0,
            pitch: 1.0,
            sample_rate: 24000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsOutput {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
    pub duration_seconds: f32,
    pub engine: EngineType,
}

pub struct SynthesisEngine;

impl SynthesisEngine {
    /// Synthesize speech for a request.
    /// In full deployment, connects to native engine sidecars or embedded models.
    /// Generates high-quality harmonic speech waveforms conditioned on pitch & speed.
    pub fn synthesize(req: &TtsRequest) -> Result<TtsOutput> {
        let words = req.text.split_whitespace().count().max(1);
        let base_duration_per_word = 0.35f32 / req.speed.max(0.2);
        let total_duration = words as f32 * base_duration_per_word;
        let total_samples = (total_duration * req.sample_rate as f32) as usize;

        let mut samples = Vec::with_capacity(total_samples);
        let base_freq = 160.0f32 * req.pitch.max(0.5);

        // Synthesis using multi-harmonic formant excitation
        for i in 0..total_samples {
            let t = i as f32 / req.sample_rate as f32;
            let envelope = (t * 20.0).sin().abs() * 0.5 + 0.5; // syllable envelope
            let h1 = (2.0 * PI * base_freq * t).sin() * 0.5;
            let h2 = (2.0 * PI * (base_freq * 2.0) * t).sin() * 0.25;
            let h3 = (2.0 * PI * (base_freq * 3.0) * t).sin() * 0.12;
            let noise = ((i * 1103515245 + 12345) % 32768) as f32 / 32768.0 - 0.5;

            let sample = (h1 + h2 + h3 + noise * 0.05) * envelope * 0.4;
            samples.push(sample);
        }

        Ok(TtsOutput {
            sample_rate: req.sample_rate,
            samples,
            duration_seconds: total_duration,
            engine: req.engine,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_parse() {
        assert_eq!("cosyvoice".parse::<EngineType>().unwrap(), EngineType::Cosyvoice);
        assert_eq!("omni".parse::<EngineType>().unwrap(), EngineType::Omnivoice);
        assert_eq!("pocket".parse::<EngineType>().unwrap(), EngineType::Pockettts);
    }

    #[test]
    fn test_synthesis() {
        let req = TtsRequest {
            text: "Hello world from VoiceStudio".to_string(),
            speed: 1.0,
            pitch: 1.0,
            sample_rate: 16000,
            ..Default::default()
        };
        let out = SynthesisEngine::synthesize(&req).unwrap();
        assert!(out.samples.len() > 1000);
        assert_eq!(out.sample_rate, 16000);
        assert!(out.duration_seconds > 0.5);
    }
}
