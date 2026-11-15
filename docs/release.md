# Distribution boundary

The GitHub candidate contains the method, runnable tools, frozen lightweight
result records, metadata, tests, documentation and supplementary source/PDF.
Real raw/processed datasets, the local final-prediction archive, working caches,
new execution outputs and upstream baseline checkouts are excluded.

MIT covers the authors' code and documentation. `THIRD_PARTY.md` and retained
notices identify source-specific conditions. Dataset access and redistribution
conditions are recorded separately. LDPGraph's upstream code and author PDF are
not bundled in the repository, wheel, source package or GitHub ZIP.

The full local artifacts directory includes the selected datasets and final
prediction archive so the old research directories are not needed to verify
saved numerical results. These local-only files do not become public merely
because they are present alongside the Git checkout. Upload the audited
candidate or reviewed tracked file list, not the entire local folder.

Before public release, resolve any outstanding rights records and author
publication details. The exact editable masters of the three active schematic
images are still unverified; the actual PNGs are retained and statistical
figures remain programmatically reproducible. The package makes no claim of
conference acceptance, an artifact badge, or successful fresh execution of all
large-network settings. Validation evidence is stored in `provenance`.

The release tag and any external data/result archive should refer to the same
source/evidence manifests. Do not force-update a published artifact version or
change a frozen checksum to accommodate an unexplained mismatch.

The public repository is English-only, including source comments, docstrings,
CLI messages and LaTeX comments. Chinese review notes are maintained outside
this checkout. `python reproduction/check_english.py` enforces the text rule
in CI; PDF text and visible figure labels are also checked for the release.
Comment translations preserve executable semantics and active manuscript text;
provenance distinguishes original-source hashes from current public-file hashes.

The root `.gitattributes` disables Git newline conversion for tracked files.
Source identities and frozen evidence are byte-hashed; automatic LF/CRLF
conversion would invalidate those checks even without a scientific change.
