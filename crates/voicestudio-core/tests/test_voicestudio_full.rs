use std::f32::consts::PI;
use std::path::PathBuf;
use tempfile::NamedTempFile;
use voicestudio_core::{
    asr::{AsrProcessor, ASR_SAMPLE_RATE},
    cloning::{VoiceCloner, EMBEDDING_DIM},
    config::VoiceStudioConfig,
    dsp::AudioDsp,
    dubbing::DubbingPipeline,
    engine::{EngineType, SynthesisEngine, TtsRequest},
    subtitles::{SubtitleCue, SubtitleParser},
};

fn generate_sine_wave(freq_hz: f32, sample_rate: u32, duration_secs: f32, amp: f32) -> Vec<f32> {
    let total_samples = (sample_rate as f32 * duration_secs) as usize;
    (0..total_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * PI * freq_hz * t).sin() * amp
        })
        .collect()
}

fn write_test_wav(samples: &[f32], sample_rate: u32) -> NamedTempFile {
    let temp_file = NamedTempFile::new().unwrap();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(temp_file.path(), spec).unwrap();
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let sample_val = (clamped * 32767.0) as i16;
        writer.write_sample(sample_val).unwrap();
    }
    writer.finalize().unwrap();
    temp_file
}

#[test]
fn test_config_defaults() {
    let cfg = VoiceStudioConfig::default();
    assert_eq!(cfg.default_sample_rate, 24000);
    assert_eq!(cfg.target_lufs, -16.0);
    assert_eq!(cfg.personas_dir, PathBuf::from("personas"));
    assert_eq!(cfg.max_time_stretch, 1.25);
    assert_eq!(cfg.min_time_stretch, 0.8);
}

#[test]
fn test_engine_types_and_parsing() {
    let engines = [
        ("cosyvoice", EngineType::Cosyvoice),
        ("omnivoice", EngineType::Omnivoice),
        ("pockettts", EngineType::Pockettts),
        ("dotstts", EngineType::Dotstts),
        ("indextts", EngineType::Indextts),
        ("supertonic", EngineType::Supertonic),
        ("audiocpp", EngineType::Audiocpp),
    ];

    for (name, expected) in engines {
        let parsed: EngineType = name.parse().unwrap();
        assert_eq!(parsed, expected);
        assert!(!parsed.to_string().is_empty());
    }

    assert!("unknown_engine".parse::<EngineType>().is_err());
}

#[test]
fn test_synthesis_multi_engine() {
    let engines = [
        EngineType::Cosyvoice,
        EngineType::Omnivoice,
        EngineType::Pockettts,
        EngineType::Dotstts,
        EngineType::Indextts,
        EngineType::Supertonic,
        EngineType::Audiocpp,
    ];

    for engine in engines {
        let req = TtsRequest {
            text: "Testing VoiceStudio native speech synthesis.".to_string(),
            engine,
            speed: 1.0,
            pitch: 1.0,
            sample_rate: 24000,
            ..Default::default()
        };

        let out = SynthesisEngine::synthesize(&req).unwrap();
        assert_eq!(out.engine, engine);
        assert_eq!(out.sample_rate, 24000);
        assert!(out.duration_seconds > 1.0);
        assert_eq!(out.samples.len(), (out.duration_seconds * 24000.0) as usize);

        // Check waveform bounds
        for &s in &out.samples {
            assert!(s >= -1.0 && s <= 1.0, "Sample out of bounds: {}", s);
        }
    }
}

#[test]
fn test_synthesis_pitch_and_speed_control() {
    let base_req = TtsRequest {
        text: "Speed and pitch variation benchmark.".to_string(),
        speed: 1.0,
        pitch: 1.0,
        sample_rate: 16000,
        ..Default::default()
    };
    let base_out = SynthesisEngine::synthesize(&base_req).unwrap();

    let fast_req = TtsRequest {
        speed: 2.0,
        ..base_req.clone()
    };
    let fast_out = SynthesisEngine::synthesize(&fast_req).unwrap();
    assert!(fast_out.duration_seconds < base_out.duration_seconds);

    let slow_req = TtsRequest {
        speed: 0.5,
        ..base_req.clone()
    };
    let slow_out = SynthesisEngine::synthesize(&slow_req).unwrap();
    assert!(slow_out.duration_seconds > base_out.duration_seconds);
}

#[test]
fn test_speaker_cloning_and_persona_extraction() {
    // Generate 1.5 seconds of harmonic vocal audio (fundamental 180Hz)
    let samples = generate_sine_wave(180.0, 16000, 1.5, 0.6);
    let temp_wav = write_test_wav(&samples, 16000);

    let persona = VoiceCloner::extract_persona("Dr. Sarah Chen", temp_wav.path()).unwrap();
    assert_eq!(persona.name, "Dr. Sarah Chen");
    assert_eq!(persona.sample_rate, 16000);
    assert_eq!(persona.embedding.len(), EMBEDDING_DIM);
    assert!(persona.duration_seconds >= 1.4);
    assert!(persona.estimated_pitch_hz > 50.0 && persona.estimated_pitch_hz < 300.0);

    // Verify embedding is unit-normalized
    let norm = persona.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4);
}

#[test]
fn test_embedding_cosine_similarity() {
    let mut emb1 = vec![0.0f32; EMBEDDING_DIM];
    let mut emb2 = vec![0.0f32; EMBEDDING_DIM];
    emb1[0] = 1.0;
    emb2[0] = 1.0;

    let sim_same = VoiceCloner::cosine_similarity(&emb1, &emb2);
    assert!((sim_same - 1.0).abs() < 1e-5);

    let mut emb3 = vec![0.0f32; EMBEDDING_DIM];
    emb3[1] = 1.0;
    let sim_ortho = VoiceCloner::cosine_similarity(&emb1, &emb3);
    assert!(sim_ortho.abs() < 1e-5);

    let mut emb_neg = vec![0.0f32; EMBEDDING_DIM];
    emb_neg[0] = -1.0;
    let sim_neg = VoiceCloner::cosine_similarity(&emb1, &emb_neg);
    assert!((sim_neg - (-1.0)).abs() < 1e-5);
}

#[test]
fn test_audio_dsp_analysis_and_normalization() {
    let samples = generate_sine_wave(440.0, 44100, 1.0, 0.2);
    let stats = AudioDsp::analyze(&samples);

    assert!(stats.rms_db < 0.0);
    assert!(stats.peak_db <= 0.0);
    assert!(stats.integrated_lufs < 0.0);
    assert!(stats.dc_offset.abs() < 0.05);

    let target_lufs = -16.0;
    let normalized = AudioDsp::normalize_loudness(&samples, target_lufs);
    assert_eq!(normalized.len(), samples.len());

    let normalized_stats = AudioDsp::analyze(&normalized);
    assert!((normalized_stats.integrated_lufs - target_lufs).abs() < 1.0);

    // Verify soft limiter prevents hard clipping
    for &s in &normalized {
        assert!(s.abs() <= 1.0);
    }
}

#[test]
fn test_audio_dsp_time_stretch() {
    let samples = generate_sine_wave(300.0, 16000, 1.0, 0.5);
    assert_eq!(samples.len(), 16000);

    // Slow down (stretch to 1.5x length) -> factor = 1/1.5 ≈ 0.6667
    let stretched = AudioDsp::time_stretch(&samples, 0.666667);
    assert_eq!(stretched.len(), 24000);

    // Speed up (compress to 0.5x length) -> factor = 2.0
    let compressed = AudioDsp::time_stretch(&samples, 2.0);
    assert_eq!(compressed.len(), 8000);
}

#[test]
fn test_audio_dsp_watermark() {
    let samples = generate_sine_wave(1000.0, 16000, 0.5, 0.5);
    let watermark_id = 0xABCD1234;
    let watermarked = AudioDsp::embed_watermark(&samples, watermark_id);

    assert_eq!(watermarked.len(), samples.len());
    // Difference should be minimal (inaudible)
    for (orig, wm) in samples.iter().zip(watermarked.iter()) {
        assert!((orig - wm).abs() < 0.001);
    }
}

#[test]
fn test_asr_resampling() {
    let orig_rate = 48000;
    let samples = generate_sine_wave(1000.0, orig_rate, 1.0, 0.8);
    assert_eq!(samples.len(), 48000);

    let resampled = AsrProcessor::resample_to_16k(&samples, orig_rate);
    assert_eq!(resampled.len(), ASR_SAMPLE_RATE as usize);
}

#[test]
fn test_asr_voice_activity_detection() {
    let mut buffer = vec![0.0f32; 16000 * 2]; // 2 seconds at 16kHz
    // Add silence in first second
    // Add speech burst in second second (16000..32000)
    for i in 16000..32000 {
        let t = i as f32 / 16000.0;
        buffer[i] = (2.0 * PI * 200.0 * t).sin() * 0.4;
    }

    let frames = AsrProcessor::detect_voice_activity(&buffer, -35.0);
    assert!(!frames.is_empty());

    // Beginning frames should not be speech
    let early_speech_count = frames[..20].iter().filter(|f| f.is_speech).count();
    assert_eq!(early_speech_count, 0);

    // Later frames should be speech
    let late_speech_count = frames[frames.len() - 30..].iter().filter(|f| f.is_speech).count();
    assert!(late_speech_count > 20);
}

#[test]
fn test_subtitles_parsing_and_serialization() {
    let raw_srt = r#"1
00:00:01,000 --> 00:00:03,500
[Narrator] In a world powered by Rust.

2
00:00:04,100 --> 00:00:07,850
Voice synthesis is fast and memory safe.
"#;

    let cues = SubtitleParser::parse_srt(raw_srt).unwrap();
    assert_eq!(cues.len(), 2);

    assert_eq!(cues[0].index, 1);
    assert_eq!(cues[0].start_ms, 1000);
    assert_eq!(cues[0].end_ms, 3500);
    assert_eq!(cues[0].duration_ms(), 2500);
    assert_eq!(cues[0].speaker.as_deref(), Some("Narrator"));
    assert_eq!(cues[0].text, "In a world powered by Rust.");

    assert_eq!(cues[1].index, 2);
    assert_eq!(cues[1].start_ms, 4100);
    assert_eq!(cues[1].end_ms, 7850);
    assert_eq!(cues[1].speaker, None);
    assert_eq!(cues[1].text, "Voice synthesis is fast and memory safe.");

    let rendered = SubtitleParser::to_srt(&cues);
    let reparsed = SubtitleParser::parse_srt(&rendered).unwrap();
    assert_eq!(reparsed.len(), 2);
    assert_eq!(reparsed[0].text, cues[0].text);
    assert_eq!(reparsed[1].text, cues[1].text);
}

#[test]
fn test_dubbing_pipeline_full_workflow() {
    let cues = vec![
        SubtitleCue {
            index: 1,
            start_ms: 500,
            end_ms: 2500,
            text: "Hello and welcome to the presentation.".to_string(),
            speaker: Some("Presenter".to_string()),
        },
        SubtitleCue {
            index: 2,
            start_ms: 3000,
            end_ms: 5500,
            text: "Today we demonstrate automated speech dubbing.".to_string(),
            speaker: Some("Presenter".to_string()),
        },
    ];

    let plans = DubbingPipeline::plan_and_render_segments(
        &cues,
        EngineType::Cosyvoice,
        "en",
        24000,
    ).unwrap();

    assert_eq!(plans.len(), 2);
    assert_eq!(plans[0].target_duration_ms, 2000);
    assert_eq!(plans[1].target_duration_ms, 2500);
    assert!(!plans[0].adjusted_speech_samples.is_empty());
    assert!(!plans[1].adjusted_speech_samples.is_empty());

    // Mix with 6 seconds of background audio
    let bg_music = generate_sine_wave(120.0, 24000, 6.0, 0.1);
    let dub_result = DubbingPipeline::mix_with_background(&plans, &bg_music, 24000, -12.0);

    assert_eq!(dub_result.total_segments, 2);
    assert_eq!(dub_result.sample_rate, 24000);
    assert!(dub_result.total_duration_seconds >= 5.5);
    assert_eq!(
        dub_result.output_samples.len(),
        (dub_result.total_duration_seconds * 24000.0) as usize
    );
}
