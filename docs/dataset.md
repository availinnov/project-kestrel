# Dataset extraction v1

`kestrel dataset-extract INPUT OUTPUT_JSON` writes a source-usage and retained-fragment
dataset. The output must be a new file. The input and external media are never modified
or accessed for metadata discovery.

Only project/catalog metadata, imported source metadata, and the main timeline and
its extra document are read. Binary previews and caches are skipped. Selected JSON
is bounded to 128 MiB per entry and 256 MiB total; writer parsing is unchanged.

Catalog IDs identify sources. Clip catalog linkage is preferred; a unique exact
source-path fallback is used only when that linkage is absent. Legacy catalog-linked
clips without a timeline resource table are supported. Unresolved ordinary clip
linkage fails explicitly. Audio pairing uses catalog/source identity, geometry and
available placement linkage. Missing or ambiguous audio produces null gain.

Usage means an ordinary video fragment exists on the active timeline. Audio-only
presence does not imply usage. Each retained fragment remains separate, including
repeated and overlapping source ranges. Retained ticks are summed, so the retained
ratio can exceed one. No trim targets or editing history are inferred.

Fragments are sorted by timeline begin, track index and clip ID. Transitions retain
identifiers and known geometry; an adjacent fragment is linked only when unique on
the same track. Unknown effect IDs and parameter names are summarized without
opaque payloads. Simple explicit numeric image controls are best effort; their raw
values are not converted into a common grading scale.

The JSON separates sources, timeline fragments, transitions and effect summaries.
The CLI prints usage, coverage, exclusions, ignored binary count, output size and
runtime. Unsupported catalog types or entries without video metadata are counted
as excluded rather than labeled unused ordinary video sources.
