use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubtitleCue {
    pub index: usize,
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    pub speaker: Option<String>,
}

impl SubtitleCue {
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    pub fn format_srt_timestamp(ms: u64) -> String {
        let hours = ms / 3_600_000;
        let mins = (ms % 3_600_000) / 60_000;
        let secs = (ms % 60_000) / 1000;
        let millis = ms % 1000;
        format!("{:02}:{:02}:{:02},{:03}", hours, mins, secs, millis)
    }

    pub fn parse_srt_timestamp(ts: &str) -> Result<u64> {
        let ts = ts.trim().replace('.', ",");
        let parts: Vec<&str> = ts.split(',').collect();
        if parts.len() != 2 {
            anyhow::bail!("Invalid timestamp format: {}", ts);
        }
        let time_parts: Vec<&str> = parts[0].split(':').collect();
        if time_parts.len() != 3 {
            anyhow::bail!("Invalid time part in timestamp: {}", ts);
        }

        let hours: u64 = time_parts[0].parse()?;
        let mins: u64 = time_parts[1].parse()?;
        let secs: u64 = time_parts[2].parse()?;
        let millis: u64 = parts[1].parse()?;

        Ok(hours * 3_600_000 + mins * 60_000 + secs * 1000 + millis)
    }
}

pub struct SubtitleParser;

impl SubtitleParser {
    /// Parse SRT subtitle file content
    pub fn parse_srt(content: &str) -> Result<Vec<SubtitleCue>> {
        let blocks = content.replace("\r\n", "\n");
        let re = Regex::new(
            r"(?m)^(\d+)\s*\n(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\n([\s\S]*?)(?:\n\n|\z)",
        )?;

        let mut cues = Vec::new();
        for cap in re.captures_iter(&blocks) {
            let index: usize = cap[1].parse().unwrap_or(cues.len() + 1);
            let start_ms = SubtitleCue::parse_srt_timestamp(&cap[2])
                .with_context(|| format!("Failed to parse start timestamp {}", &cap[2]))?;
            let end_ms = SubtitleCue::parse_srt_timestamp(&cap[3])
                .with_context(|| format!("Failed to parse end timestamp {}", &cap[3]))?;
            let raw_text = cap[4].trim();

            let (speaker, text) = if let Some(stripped) = raw_text.strip_prefix('[') {
                if let Some(close_idx) = stripped.find(']') {
                    let spk = &stripped[..close_idx];
                    let rest = stripped[close_idx + 1..].trim();
                    (Some(spk.to_string()), rest.to_string())
                } else {
                    (None, raw_text.to_string())
                }
            } else {
                (None, raw_text.to_string())
            };

            cues.push(SubtitleCue {
                index,
                start_ms,
                end_ms,
                text,
                speaker,
            });
        }

        Ok(cues)
    }

    /// Render subtitle cues to SRT format string
    pub fn to_srt(cues: &[SubtitleCue]) -> String {
        let mut out = String::new();
        for (i, cue) in cues.iter().enumerate() {
            let index = if cue.index > 0 { cue.index } else { i + 1 };
            out.push_str(&format!("{}\n", index));
            out.push_str(&format!(
                "{} --> {}\n",
                SubtitleCue::format_srt_timestamp(cue.start_ms),
                SubtitleCue::format_srt_timestamp(cue.end_ms)
            ));
            if let Some(spk) = &cue.speaker {
                out.push_str(&format!("[{}] {}\n\n", spk, cue.text));
            } else {
                out.push_str(&format!("{}\n\n", cue.text));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srt_roundtrip() {
        let srt_input = "1\n00:00:01,000 --> 00:00:03,500\n[Alice] Welcome to VoiceStudio.\n\n2\n00:00:04,000 --> 00:00:06,000\nFully-local ElevenLabs alternative.\n\n";
        let cues = SubtitleParser::parse_srt(srt_input).unwrap();
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].start_ms, 1000);
        assert_eq!(cues[0].end_ms, 3500);
        assert_eq!(cues[0].speaker.as_deref(), Some("Alice"));

        let rendered = SubtitleParser::to_srt(&cues);
        assert!(rendered.contains("00:00:01,000 --> 00:00:03,500"));
        assert!(rendered.contains("[Alice] Welcome to VoiceStudio."));
    }
}
