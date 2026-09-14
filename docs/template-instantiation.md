# Empty template instantiation

This controlled experiment creates a new empty instance from a validated template.
It refreshes project and timeline identities, their known references, creation
timestamps, the project name, and the output path. Optional width, height, and
integer frame rate overrides update both copies of the settings.

The `project_source` value is treated as an instance-specific opaque identifier.
No derivation has been established; a fresh independent 32-character hexadecimal
token is generated. It is not derived from the project UUID.

The two verified catalog timeline UUID payloads are rewritten: user data key 11000
contains 38 ASCII UUID bytes followed by 26 NUL bytes; key 30309 contains the same
38 ASCII bytes without padding. Both references must agree with the catalog.
Key 3 holds an independent braced ASCII UUID with 26 NUL padding bytes (64 bytes
total). Key 140 holds a 32-byte lowercase ASCII hexadecimal token. Both vary
across independent empty templates and receive fresh values. No derivation for
key 140 has been established; it is treated as an opaque token, not a checksum.
Malformed or ambiguous known payloads prevent writing.

The first audio bus receives a fresh UUID, and every matching track bus reference
is updated. Other buses, track identities, unrelated user data, and stable asset
identifiers remain unchanged. All track bus references must resolve uniquely.
The observed serial number equals the numeric timeline ID plus one. Both numeric
values are preserved: the samples do not establish a need to regenerate them.
Post-write validation checks payload structure, fresh identities, and bus links.
These checks do not establish compatibility with an external application.

All children of the main timeline media directory move to its fresh identity.
Unknown children and optional thumbnails are retained. JSON edits preserve
surrounding text. Entry contents outside the targeted edits are preserved exactly;
ZIP compression streams may differ after repacking. Archive comments, entry
compression methods, timestamps, and metadata are retained where supported.

The output must be new. It is written to a temporary file, reparsed, and validated
before publication. No external source files are accessed or generated.
