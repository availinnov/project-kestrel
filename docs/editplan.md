# EditPlan v1

`kestrel project-apply-plan INPUT PLAN_JSON OUTPUT` applies source decisions to
an existing project using its ordinary template pair and destination tracks.
The output must be a new file.

```json
{
  "version": 1,
  "clips": [
    {
      "catalog_id": "source-id",
      "keep": true,
      "trim_start_ticks": 10000000,
      "trim_end_ticks": 0
    }
  ]
}
```

Every eligible source must appear exactly once. Unknown fields, missing sources,
ineligible sources, invalid types and duplicate IDs are rejected. Dropped sources
require zero trims. At least one source must be kept.

Plan entry order does not control placement. Kept sources follow the established
imported creation-time order, with deterministic name and catalog-ID tie-breakers.
Dropped catalog-only sources do not create timeline resources or placements.

Trims remove ticks from the beginning and end of the full frame-aligned source
range. There are 10,000,000 ticks per second. Requested boundaries snap to the
nearest timeline frame, ties upward, then to integer ticks with the same rule.
All frame decisions use exact rational arithmetic. Collapsed or invalid ranges
are rejected. Kept ranges are packed contiguously from zero.

Reports include requested trims, resulting ranges, snapping adjustments, dropped
sources, generated identities and validation results. The existing sequence
command remains available for all-keep, zero-trim generation.
