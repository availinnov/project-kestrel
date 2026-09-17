# Rules and media analysis v1

`kestrel media-analyze INPUT OUTPUT_JSON --rules rules/default_rules_v1.json`
measures imported local video sources and emits proposed actions. Omitting `--rules`
produces measurements only. It does not edit projects or generate plans.

Decoding requires an external `ffmpeg` executable on PATH, or
`--ffmpeg C:/tools/ffmpeg.exe`. No new Python dependency is required. Missing media
and decoder failures are recorded per source. Duration-only actions can still be
proposed when decoding is unavailable. Network URLs and network paths are rejected.
The output must be a new file.

## Measurements

Duration uses positive imported duration ticks divided by 10,000,000. Sparse video
sampling uses midpoint timestamps distributed over the entire duration, at most
24 frames and nominally one per second. Each seek decodes one 32x32 grayscale frame;
frame mean luma is normalized to 0..1. Seeking avoids decoding every frame, although
compressed media may require decoding from a preceding keyframe. Each sample starts
a separate decoder process; this favors bounded sparse decoding over throughput.

CLI settings: `--max-frames` (1..128), `--sample-rate` (samples per second),
`--black-luma-threshold` (default 0.05), and `--high-peak-dbfs` (default -3).
The Python DetectorConfig also exposes a timeout (300 seconds per decoder process)
and audio window duration (0.1 seconds). Detector thresholds define measurements;
rule thresholds define proposed decisions.

Audio is streamed as mono 16kHz signed PCM in bounded windows. RMS and sample peak
are reported in dBFS. High-peak count is the number of windows reaching the configured
sample-peak threshold; ratio is high windows divided by all windows, including a
final partial window. These are window counts, not deduplicated acoustic events.
Downmixing and resampling can alter peaks; this is not native multichannel true peak.

`integrated_lufs` is always null in v1. RMS is not LUFS and is explicitly named
`rms_dbfs`. Silence has `silent: true` and null dBFS values (negative infinity cannot
be represented in standard JSON). No audio yields `available: false`; failed audio
yields `available: null`. A successful video measurement survives audio failure.

## Strict rule specification

The root contains exactly `version: 1` and a `rules` array. Every rule requires
`rule_id`, boolean `enabled`, `mode` (`auto` or `review`), integer `priority`,
`when`, and `then`. IDs must be unique. Unknown fields and duplicate JSON keys fail.

Conditions are either nonempty `all`/`any` arrays of conditions or a leaf with exactly
`signal`, `op`, and `value`. Dotted signal paths support nested measurements. Operators
are `<`, `<=`, `>`, `>=`, `==`, and `!=`. Ordered comparisons require finite numbers.
Missing/null signals never match, including `!=`; booleans are not numeric values.
No expressions, code, or scripts are accepted. Nesting is limited to 16 levels.

`then` is a nonempty object with one or more of:

- `"drop": true`
- `"color_tag": 4` (stored numeric value 1..13)
- `"warning": "code"`
- `"normalize_audio": {"target_lufs": -18}`

Actions carry rule ID and mode. Enabled rules run by descending priority, then rule
ID for ties. All matching actions are retained, including review actions alongside
a drop. Exact duplicates including provenance are deduplicated.

## Starter policies and limits

Default rules propose dropping durations below 2 seconds and black ratios above
0.50, and reviewing durations of at least 20 seconds with numeric tag 2.
Audio review uses RMS between -30 and -12 dBFS with at least one high-peak window and
high-peak ratio at most 0.10, proposing tag 4 and a warning. Quiet RMS below -30 dBFS
(or digital silence) and loud RMS above -12 dBFS propose normalization to -18 LUFS.
These RMS thresholds are provisional proxies, not LUFS estimates. A target is only
a proposal: no gain is computed or applied, and silence cannot be repaired by gain.
All thresholds require tuning against real measurements; they are never auto-tuned.

Analysis includes configuration, signals, failures, actions and summary distributions.
`analyzed_count` counts sources with decoded video or audio; metadata-only duration
results do not count. Partial decode failures count in `failed_count`. Empty measured
distributions have count zero and null percentiles; zero matching cases must not be
interpreted as evidence of absence when no decoding succeeded.
