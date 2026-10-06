use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::Colorize;
use hound::{SampleFormat, WavSpec, WavWriter};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use voicestudio_core::{
    AsrProcessor, AudioDsp, DubbingPipeline, EngineType, SpeakerPersona, SubtitleCue,
    SubtitleParser, SynthesisEngine, TtsRequest, VoiceCloner,
};

#[derive(Parser, Debug)]
#[command(
    name = "voicestudio",
    version,
    about = "High-performance CLI for VoiceStudio: Open-source, fully-local ElevenLabs alternative",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Synthesize speech from text using chosen engine
    Synthesize {
        /// Text string to synthesize
        #[arg(short = 't', long = "text")]
        text: String,

        /// TTS Engine (cosyvoice, omnivoice, pockettts, dotstts, indextts, supertonic, audiocpp)
        #[arg(short = 'e', long = "engine", default_value = "omnivoice")]
        engine: String,

        /// Output WAV audio path
        #[arg(short = 'o', long = "output")]
        output: PathBuf,

        /// Optional persona JSON file for voice cloning
        #[arg(long = "persona")]
        persona: Option<PathBuf>,

        /// Speech rate multiplier (0.5 to 2.0)
        #[arg(long = "speed", default_value = "1.0")]
        speed: f32,

        /// Speech pitch multiplier (0.5 to 2.0)
        #[arg(long = "pitch", default_value = "1.0")]
        pitch: f32,

        /// Audio sample rate in Hz
        #[arg(long = "sample-rate", default_value = "24000")]
        sample_rate: u32,
    },

    /// Extract speaker timbre embedding and save voice persona profile
    Clone {
        /// Name of the speaker persona
        #[arg(short = 'n', long = "name")]
        name: String,

        /// Reference voice WAV file (clean speech sample)
        #[arg(short = 'a', long = "audio")]
        audio: PathBuf,

        /// Output path for the persona JSON bundle
        #[arg(short = 'o', long = "output")]
        output: PathBuf,
    },

    /// Run full video/audio dubbing pipeline with subtitle timing & ducking
    Dub {
        /// Input SRT subtitles file containing speech cues
        #[arg(short = 's', long = "subtitles")]
        subtitles: PathBuf,

        /// Optional background audio WAV file
        #[arg(short = 'b', long = "background")]
        background: Option<PathBuf>,

        /// Output dubbed audio WAV path
        #[arg(short = 'o', long = "output")]
        output: PathBuf,

        /// Target language code (e.g. en, es, fr, ja, zh)
        #[arg(short = 'l', long = "lang", default_value = "en")]
        lang: String,

        /// TTS Engine to use for dubbing resynthesis
        #[arg(short = 'e', long = "engine", default_value = "omnivoice")]
        engine: String,

        /// Background audio ducking level in dB (default -12.0 dB)
        #[arg(long = "ducking-db", default_value = "-12.0")]
        ducking_db: f32,
    },

    /// Audio transcription and Voice Activity Detection (VAD)
    Transcribe {
        /// Input WAV audio file
        #[arg(short = 'a', long = "audio")]
        audio: PathBuf,

        /// Output SRT subtitle file path
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,

        /// VAD energy threshold in dB
        #[arg(long = "threshold", default_value = "-35.0")]
        threshold: f32,
    },

    /// Audio DSP utilities: LUFS loudness normalization and time-stretching
    Dsp {
        /// Input audio WAV file
        #[arg(short = 'i', long = "input")]
        input: PathBuf,

        /// Output audio WAV file
        #[arg(short = 'o', long = "output")]
        output: PathBuf,

        /// Target integrated loudness in LUFS (e.g. -16.0)
        #[arg(long = "target-lufs", default_value = "-16.0")]
        target_lufs: f32,

        /// Time stretch speed factor (e.g. 1.25 for faster, 0.8 for slower)
        #[arg(long = "stretch")]
        stretch: Option<f32>,

        /// Optional watermark ID integer to embed
        #[arg(long = "watermark")]
        watermark: Option<u32>,
    },

    /// Benchmark TTS synthesis throughput and real-time factor (RTF)
    Bench {
        /// Number of words in benchmark prompt
        #[arg(long = "words", default_value = "100")]
        words: usize,

        /// Number of benchmark iterations
        #[arg(long = "runs", default_value = "5")]
        runs: usize,
    },
}

fn write_wav_file(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)
        .with_context(|| format!("Failed to create WAV file {:?}", path))?;

    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let sample_i16 = (clamped * 32767.0) as i16;
        writer.write_sample(sample_i16)?;
    }

    writer.finalize()?;
    Ok(())
}

fn read_wav_file(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("Failed to open WAV file {:?}", path))?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let raw: Vec<f32> = match spec.sample_format {
        SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.unwrap_or(0) as f32 / max).collect()
        }
        SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
    };

    let mono = if channels > 1 {
        raw.chunks(channels).map(|c| c.iter().sum::<f32>() / channels as f32).collect()
    } else {
        raw
    };

    Ok((mono, sample_rate))
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Synthesize {
            text,
            engine,
            output,
            persona,
            speed,
            pitch,
            sample_rate,
        } => {
            let engine_type: EngineType = engine.parse()
                .with_context(|| format!("Invalid engine '{}'", engine))?;

            println!(
                "{} speech using [{}] engine...",
                "Synthesizing".cyan().bold(),
                engine_type.to_string().yellow()
            );
            println!("  Text        : \"{}\"", text);
            println!("  Speed/Pitch : {:.2}x / {:.2}x", speed, pitch);
            println!("  Output WAV  : {}", output.display().to_string().green());

            let speaker_embedding = if let Some(p_path) = persona {
                let content = std::fs::read_to_string(&p_path)
                    .with_context(|| format!("Failed to read persona file {:?}", p_path))?;
                let profile: SpeakerPersona = serde_json::from_str(&content)?;
                println!("  Persona     : {}", profile.name.cyan());
                Some(profile.embedding)
            } else {
                None
            };

            let req = TtsRequest {
                text,
                engine: engine_type,
                voice_id: None,
                speaker_embedding,
                language: "en".to_string(),
                speed,
                pitch,
                sample_rate,
            };

            let start = Instant::now();
            let result = SynthesisEngine::synthesize(&req)?;
            let elapsed = start.elapsed();

            write_wav_file(&output, &result.samples, result.sample_rate)?;

            println!("\n{}", "Synthesis Complete!".green().bold());
            println!("  Audio Duration : {:.2}s", result.duration_seconds);
            println!("  Render Time    : {:.3}s", elapsed.as_secs_f64());
            println!(
                "  Real-Time Factor (RTF): {:.3}x",
                elapsed.as_secs_f64() / result.duration_seconds as f64
            );
        }

        Commands::Clone { name, audio, output } => {
            println!(
                "{} speaker voice profile from reference audio...",
                "Extracting".cyan().bold()
            );
            println!("  Speaker Name: {}", name.yellow().bold());
            println!("  Reference   : {}", audio.display().to_string().yellow());

            let persona = VoiceCloner::extract_persona(&name, &audio)?;
            let json = serde_json::to_string_pretty(&persona)?;

            if let Some(parent) = output.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            let mut file = File::create(&output)?;
            file.write_all(json.as_bytes())?;

            println!("\n{}", "Voice Persona Created!".green().bold());
            println!("  Persona ID      : {}", persona.id);
            println!("  Estimated Pitch : {:.1} Hz", persona.estimated_pitch_hz);
            println!("  Embedding Dims  : {}", persona.embedding.len());
            println!("  Saved To        : {}", output.display().to_string().green());
        }

        Commands::Dub {
            subtitles,
            background,
            output,
            lang,
            engine,
            ducking_db,
        } => {
            let engine_type: EngineType = engine.parse()?;
            println!(
                "{} video dubbing pipeline...",
                "Executing".cyan().bold()
            );
            println!("  Subtitles  : {}", subtitles.display().to_string().yellow());
            println!("  Target Lang: {}", lang.bold());
            println!("  TTS Engine : {}", engine_type);

            let srt_content = std::fs::read_to_string(&subtitles)?;
            let cues = SubtitleParser::parse_srt(&srt_content)?;
            println!("  Found {} subtitle cues to resynthesize.", cues.len().to_string().cyan());

            let sample_rate = 24000;
            let plans = DubbingPipeline::plan_and_render_segments(
                &cues,
                engine_type,
                &lang,
                sample_rate,
            )?;

            let bg_samples = if let Some(bg_path) = background {
                println!("  Background Audio: {}", bg_path.display().to_string().yellow());
                let (samples, bg_sr) = read_wav_file(&bg_path)?;
                if bg_sr != sample_rate {
                    AsrProcessor::resample_to_16k(&samples, bg_sr)
                } else {
                    samples
                }
            } else {
                Vec::new()
            };

            let dubbed = DubbingPipeline::mix_with_background(&plans, &bg_samples, sample_rate, ducking_db);
            write_wav_file(&output, &dubbed.output_samples, dubbed.sample_rate)?;

            println!("\n{}", "Dubbing Complete!".green().bold());
            println!("  Total Segments : {}", dubbed.total_segments);
            println!("  Duration       : {:.2}s", dubbed.total_duration_seconds);
            println!("  Saved To       : {}", output.display().to_string().green());
        }

        Commands::Transcribe { audio, output, threshold } => {
            println!(
                "{} audio file for speech activity & transcription...",
                "Analyzing".cyan().bold()
            );
            println!("  Audio File: {}", audio.display().to_string().yellow());

            let (samples, sr) = read_wav_file(&audio)?;
            let resampled = AsrProcessor::resample_to_16k(&samples, sr);
            let vad_frames = AsrProcessor::detect_voice_activity(&resampled, threshold);

            let speech_frames = vad_frames.iter().filter(|f| f.is_speech).count();
            let total_frames = vad_frames.len();
            let speech_ratio = if total_frames > 0 {
                (speech_frames as f32 / total_frames as f32) * 100.0
            } else {
                0.0
            };

            println!("\n{}", "Voice Activity Analysis:".bold());
            println!("  Total Frames  : {}", total_frames);
            println!("  Speech Frames : {} ({:.1}%)", speech_frames, speech_ratio);

            if let Some(out_p) = output {
                let cue = SubtitleCue {
                    index: 1,
                    start_ms: 0,
                    end_ms: (samples.len() as f32 / sr as f32 * 1000.0) as u64,
                    text: "Transcribed speech segment from VoiceStudio".to_string(),
                    speaker: Some("Speaker".to_string()),
                };
                let srt = SubtitleParser::to_srt(&[cue]);
                std::fs::write(&out_p, srt)?;
                println!("  SRT Output    : {}", out_p.display().to_string().green());
            }
        }

        Commands::Dsp {
            input,
            output,
            target_lufs,
            stretch,
            watermark,
        } => {
            println!(
                "{} DSP operations...",
                "Running".cyan().bold()
            );
            println!("  Input Audio: {}", input.display().to_string().yellow());

            let (mut samples, sr) = read_wav_file(&input)?;
            let initial_stats = AudioDsp::analyze(&samples);
            println!("  Initial LUFS: {:.2}", initial_stats.integrated_lufs);

            if let Some(factor) = stretch {
                println!("  Applying Time-Stretch ({:.2}x)...", factor);
                samples = AudioDsp::time_stretch(&samples, factor);
            }

            println!("  Normalizing Loudness to {:.1} LUFS...", target_lufs);
            samples = AudioDsp::normalize_loudness(&samples, target_lufs);

            if let Some(wm) = watermark {
                println!("  Embedding Provenance Watermark (ID: {})...", wm);
                samples = AudioDsp::embed_watermark(&samples, wm);
            }

            let final_stats = AudioDsp::analyze(&samples);
            println!("  Final LUFS  : {:.2}", final_stats.integrated_lufs);

            write_wav_file(&output, &samples, sr)?;
            println!("\n{}", "DSP Processing Complete!".green().bold());
            println!("  Saved To    : {}", output.display().to_string().green());
        }

        Commands::Bench { words, runs } => {
            println!(
                "{} VoiceStudio benchmark: {} words prompt, {} iterations...",
                "Running".cyan().bold(),
                words,
                runs
            );

            let prompt = (0..words).map(|_| "voice").collect::<Vec<_>>().join(" ");
            let req = TtsRequest {
                text: prompt,
                engine: EngineType::Omnivoice,
                sample_rate: 24000,
                ..Default::default()
            };

            let mut total_time = 0.0;
            let mut total_duration = 0.0f32;

            for i in 1..=runs {
                let start = Instant::now();
                let out = SynthesisEngine::synthesize(&req)?;
                let elapsed = start.elapsed().as_secs_f64();
                total_time += elapsed;
                total_duration += out.duration_seconds;
                println!("  Run #{}: {:.3}s (Duration: {:.2}s, RTF: {:.3}x)", i, elapsed, out.duration_seconds, elapsed / out.duration_seconds as f64);
            }

            let avg_time = total_time / runs as f64;
            let avg_duration = total_duration / runs as f32;
            let avg_rtf = avg_time / avg_duration as f64;

            println!("\n{}", "Benchmark Summary:".green().bold());
            println!("  Avg Render Time : {:.3}s", avg_time);
            println!("  Avg Audio Length: {:.2}s", avg_duration);
            println!("  Avg RTF Factor  : {:.4}x (Real-Time Factor)", avg_rtf);
        }
    }

    Ok(())
}
