# Experimental sequence builder

`kestrel project-build-sequence INPUT OUTPUT --count N` places the first N
eligible, distinct catalog sources consecutively from timeline zero. It requires
ordinary video with audio and a supported existing AV template pair. Each template
track must contain exactly its template clip. The generated sequence replaces that
pair on the same tracks. All other tracks stay unchanged; unrelated content
extending beyond the generated sequence causes a conservative failure.

Sources are sorted by the positive integer `sourceInfo.basicInfo.createDate` in
imported `media.json`, then display name (case-folded, then original), then catalog
ID. Missing or invalid capture times sort last using the same name/ID fallback
and are reported. The template source participates normally. Existing timeline
resources are reused.
Catalog-only sources require complete imported metadata and a supported Resource
template; unresolved source fields are reported instead of fabricated. No source
files are accessed. No catalog entries or media assets are created.

Source endpoints use exact rational arithmetic: round duration to the nearest
timeline frame, ties up, then round the rational endpoint to integer ticks, ties
up. Every generated AV pair shares geometry and a dynamic constant-speed payload
with the verified opaque MD5 constant. No checksum derivation is attempted.
Video offsetEnd follows an existing full-source representation when available,
otherwise the supported template's representation; audio uses the aligned end.

Fresh placement, clip and effect instance identities come from the shared writer
allocator. Source identities are generated only for new timeline resources.
Encoded source paths and linkage fields are patched by structural location.
Default effect-local fields follow the existing clone writer's preservation rules.

The output must be a new file. Post-write checks reparse the archive, verify
generated geometry, resources and placement links, and compare against the exact
allowed structural changes. Existing project identities, timestamps, source
catalog items, and unrelated entry contents are preserved. ZIP timestamps are
retained; compression streams, flags, sizes, CRCs and offsets can differ on repack.

The normal CLI report lists selected sources, generated identities, destination
tracks, durations and modified entries. Internal validation does not establish
external application compatibility; playback, audio waveform, trim/split and
save/reopen still need manual validation of generated outputs.
