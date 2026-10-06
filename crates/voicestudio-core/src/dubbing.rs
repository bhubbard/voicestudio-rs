use crate::dsp::AudioDsp;
use crate::engine::{EngineType, SynthesisEngine, TtsRequest};
use crate::subtitles::SubtitleCue;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DubbingSegmentPlan {
    pub cue_index: usize,
    pub start_ms: u64,
    pub target_duration_ms: u64,
    pub synthesized_duration_ms: u64,
    pub stretch_factor: f32,
    pub adjusted_speech_samples: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DubbingResult {
    pub total_segments: usize,
    pub total_duration_seconds: f32,
    pub output_samples: Vec<f32>,
    pub sample_rate: u32,
}

pub struct DubbingPipeline;

impl DubbingPipeline {
    /// Plan duration and fit synthesized dialogue into video subtitle cues
    pub fn plan_and_render_segments(
        cues: &[SubtitleCue],
        engine: EngineType,
        target_lang: &str,
        sample_rate: u32,
    ) -> Result<Vec<DubbingSegmentPlan>> {
        let mut plans = Vec::new();

        for cue in cues {
            let target_duration_ms = cue.duration_ms();
            let req = TtsRequest {
                text: cue.text.clone(),
                engine,
                language: target_lang.to_string(),
                sample_rate,
                ..Default::default()
            };

            let synth = SynthesisEngine::synthesize(&req)
                .with_context(|| format!("Failed synthesizing cue {}", cue.index))?;

            let synth_duration_ms = (synth.duration_seconds * 1000.0) as u64;

            // Compute stretch factor: if synth is 2000ms but video window is 2500ms, stretch_factor = 2000 / 2500 = 0.8
            let stretch_factor = if target_duration_ms > 0 && synth_duration_ms > 0 {
                (synth_duration_ms as f32 / target_duration_ms as f32).clamp(0.75, 1.35)
            } else {
                1.0
            };

            let adjusted_speech = AudioDsp::time_stretch(&synth.samples, stretch_factor);

            plans.push(DubbingSegmentPlan {
                cue_index: cue.index,
                start_ms: cue.start_ms,
                target_duration_ms,
                synthesized_duration_ms: synth_duration_ms,
                stretch_factor,
                adjusted_speech_samples: adjusted_speech,
            });
        }

        Ok(plans)
    }

    /// Mix dubbed speech dialogue with background music/effects using audio ducking
    pub fn mix_with_background(
        plans: &[DubbingSegmentPlan],
        background_samples: &[f32],
        sample_rate: u32,
        ducking_db: f32,
    ) -> DubbingResult {
        let ducking_gain = 10.0f32.powf(ducking_db / 20.0); // e.g. -12dB -> ~0.25

        // Determine total duration in samples
        let max_segment_end_ms = plans
            .iter()
            .map(|p| p.start_ms + (p.adjusted_speech_samples.len() as f32 / sample_rate as f32 * 1000.0) as u64)
            .max()
            .unwrap_or(0);

        let total_samples = ((max_segment_end_ms as f32 / 1000.0) * sample_rate as f32) as usize;
        let final_len = total_samples.max(background_samples.len());

        let mut output = vec![0.0f32; final_len];

        // Copy background audio with ducking mask
        let mut speech_mask = vec![false; final_len];

        for plan in plans {
            let start_idx = ((plan.start_ms as f32 / 1000.0) * sample_rate as f32) as usize;
            for i in 0..plan.adjusted_speech_samples.len() {
                let out_idx = start_idx + i;
                if out_idx < final_len {
                    speech_mask[out_idx] = true;
                    output[out_idx] += plan.adjusted_speech_samples[i];
                }
            }
        }

        // Add background track with ducking
        for i in 0..final_len {
            let bg = if i < background_samples.len() { background_samples[i] } else { 0.0 };
            let gain = if speech_mask[i] { ducking_gain } else { 1.0 };
            output[i] += bg * gain;
        }

        // Normalize output loudness
        let normalized = AudioDsp::normalize_loudness(&output, -16.0);

        DubbingResult {
            total_segments: plans.len(),
            total_duration_seconds: final_len as f32 / sample_rate as f32,
            output_samples: normalized,
            sample_rate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dubbing_pipeline() {
        let cues = vec![
            SubtitleCue {
                index: 1,
                start_ms: 1000,
                end_ms: 3000,
                text: "First dubbed sentence.".to_string(),
                speaker: Some("Alice".to_string()),
            },
        ];

        let plans = DubbingPipeline::plan_and_render_segments(
            &cues,
            EngineType::Omnivoice,
            "en",
            16000,
        ).unwrap();

        assert_eq!(plans.len(), 1);
        assert!(!plans[0].adjusted_speech_samples.is_empty());

        let bg = vec![0.05f32; 16000 * 5]; // 5 seconds of background
        let result = DubbingPipeline::mix_with_background(&plans, &bg, 16000, -12.0);
        assert_eq!(result.total_segments, 1);
        assert!(result.total_duration_seconds >= 3.0);
    }
}
