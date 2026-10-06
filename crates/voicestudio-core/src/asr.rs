use serde::{Deserialize, Serialize};

pub const ASR_SAMPLE_RATE: u32 = 16000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrFrame {
    pub frame_idx: usize,
    pub is_speech: bool,
    pub energy_db: f32,
}

pub struct AsrProcessor;

impl AsrProcessor {
    /// Resample audio to 16 kHz for ASR engines (Whisper, Sherpa)
    pub fn resample_to_16k(input: &[f32], src_rate: u32) -> Vec<f32> {
        if src_rate == ASR_SAMPLE_RATE || input.is_empty() {
            return input.to_vec();
        }
        let ratio = ASR_SAMPLE_RATE as f64 / src_rate as f64;
        let out_len = (input.len() as f64 * ratio).round() as usize;
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src_idx = i as f64 / ratio;
            let low = src_idx.floor() as usize;
            let high = (low + 1).min(input.len().saturating_sub(1));
            let frac = (src_idx - low as f64) as f32;
            out.push(input[low] * (1.0 - frac) + input[high] * frac);
        }
        out
    }

    /// Run Voice Activity Detection (VAD) using energy thresholding
    /// Window size: 25ms (400 samples at 16kHz), Hop size: 10ms (160 samples)
    pub fn detect_voice_activity(samples: &[f32], threshold_db: f32) -> Vec<AsrFrame> {
        let win_size = 400;
        let hop_size = 160;
        let mut frames = Vec::new();

        if samples.len() < win_size {
            return frames;
        }

        let num_frames = (samples.len() - win_size) / hop_size + 1;
        for f in 0..num_frames {
            let start = f * hop_size;
            let end = start + win_size;
            let slice = &samples[start..end];

            let sum_sq: f32 = slice.iter().map(|&x| x * x).sum();
            let rms = (sum_sq / win_size as f32).sqrt();
            let energy_db = if rms > 1e-7 { 20.0 * rms.log10() } else { -100.0 };
            let is_speech = energy_db > threshold_db;

            frames.push(AsrFrame {
                frame_idx: f,
                is_speech,
                energy_db,
            });
        }

        frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asr_resample() {
        let samples = vec![0.5; 32000]; // 1s at 32kHz
        let resampled = AsrProcessor::resample_to_16k(&samples, 32000);
        assert_eq!(resampled.len(), 16000);
    }

    #[test]
    fn test_vad_detection() {
        let mut samples = vec![0.0; 16000];
        // Inject speech-like burst
        for i in 4000..8000 {
            samples[i] = 0.5;
        }
        let frames = AsrProcessor::detect_voice_activity(&samples, -30.0);
        assert!(!frames.is_empty());
        assert!(frames.iter().any(|f| f.is_speech));
        assert!(frames.iter().any(|f| !f.is_speech));
    }
}
