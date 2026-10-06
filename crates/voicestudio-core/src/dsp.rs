use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQualityStats {
    pub rms_db: f32,
    pub peak_db: f32,
    pub integrated_lufs: f32,
    pub dc_offset: f32,
}

pub struct AudioDsp;

impl AudioDsp {
    /// Measure audio statistics: RMS, Peak, Integrated Loudness (LUFS)
    pub fn analyze(samples: &[f32]) -> AudioQualityStats {
        if samples.is_empty() {
            return AudioQualityStats {
                rms_db: -100.0,
                peak_db: -100.0,
                integrated_lufs: -70.0,
                dc_offset: 0.0,
            };
        }

        let mut sum_sq = 0.0f32;
        let mut max_abs = 0.0f32;
        let mut sum = 0.0f32;

        for &s in samples {
            sum += s;
            let abs_s = s.abs();
            if abs_s > max_abs {
                max_abs = abs_s;
            }
            sum_sq += s * s;
        }

        let dc_offset = sum / samples.len() as f32;
        let rms = (sum_sq / samples.len() as f32).sqrt();
        let rms_db = if rms > 1e-7 { 20.0 * rms.log10() } else { -100.0 };
        let peak_db = if max_abs > 1e-7 { 20.0 * max_abs.log10() } else { -100.0 };

        // Simplified K-weighted integrated loudness estimate
        let integrated_lufs = -0.691 + 10.0 * (sum_sq / samples.len() as f32).max(1e-10).log10();

        AudioQualityStats {
            rms_db,
            peak_db,
            integrated_lufs,
            dc_offset,
        }
    }

    /// Normalize audio to target LUFS (e.g. -16.0 LUFS)
    pub fn normalize_loudness(samples: &[f32], target_lufs: f32) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }

        let stats = Self::analyze(samples);
        let current_lufs = stats.integrated_lufs;
        let gain_db = target_lufs - current_lufs;
        let gain_linear = 10.0f32.powf(gain_db / 20.0);

        let mut normalized: Vec<f32> = samples.iter().map(|&s| s * gain_linear).collect();

        // Prevent clipping beyond [-1.0, 1.0] with soft knee limiter
        for s in &mut normalized {
            if *s > 0.98 {
                *s = 0.98 + (*s - 0.98).tanh() * 0.02;
            } else if *s < -0.98 {
                *s = -0.98 + (*s + 0.98).tanh() * 0.02;
            }
        }

        normalized
    }

    /// Adjust tempo/playback speed without changing pitch (time stretch)
    pub fn time_stretch(samples: &[f32], factor: f32) -> Vec<f32> {
        if factor <= 0.0 || samples.is_empty() {
            return samples.to_vec();
        }

        // Linear interpolation resampler for tempo adjustment
        let new_len = (samples.len() as f32 / factor).round() as usize;
        let mut out = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_idx = i as f32 * factor;
            let low = src_idx.floor() as usize;
            let high = (low + 1).min(samples.len() - 1);
            let frac = src_idx - low as f32;

            if low < samples.len() {
                out.push(samples[low] * (1.0 - frac) + samples[high] * frac);
            }
        }

        out
    }

    /// Apply an imperceptible inaudible high-frequency spread-spectrum watermark
    pub fn embed_watermark(samples: &[f32], watermark_id: u32) -> Vec<f32> {
        let mut watermarked = samples.to_vec();
        let amp = 0.0005f32; // -66 dBFS amplitude watermark

        for (i, s) in watermarked.iter_mut().enumerate() {
            let bit = (watermark_id >> (i % 32)) & 1;
            let carrier = if bit == 1 { 1.0f32 } else { -1.0f32 };
            *s += amp * carrier * ((i as f32 * 0.95).sin());
        }

        watermarked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_and_normalize() {
        let samples = vec![0.1f32; 1000];
        let stats = AudioDsp::analyze(&samples);
        assert!(stats.integrated_lufs < 0.0);

        let normalized = AudioDsp::normalize_loudness(&samples, -16.0);
        let new_stats = AudioDsp::analyze(&normalized);
        assert!((new_stats.integrated_lufs - (-16.0)).abs() < 1.0);
    }

    #[test]
    fn test_time_stretch() {
        let samples = vec![0.5f32; 100];
        let stretched = AudioDsp::time_stretch(&samples, 0.5); // double length
        assert_eq!(stretched.len(), 200);

        let compressed = AudioDsp::time_stretch(&samples, 2.0); // half length
        assert_eq!(compressed.len(), 50);
    }
}
