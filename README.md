# PrivTDD

Artifacts for **PrivTDD: Triangle-Degree Distribution Estimation under
Edge-Local Differential Privacy**.

The repository contains an independently installable estimator, the exact
lightweight evidence used by the current manuscript, dataset preparation,
baseline adapters, and separate commands for figure reproduction and fresh
experiments. The implementation is a centralized simulator of the local
protocol; inference uses privatized reports and evaluation uses ground truth.

## Install

Use Python 3.12 and run from this checkout:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -c environment/constraints-windows-py312.txt -e ".[reproduce]"
python -m privtdd doctor
```

The constraints record the tested migration environment, not an inferred lock
file from historical runs. The manuscript reports Python 3.12.9; migration
verification used Python 3.12.7. See [environment details](environment/README.md).

## Reproduce the paper tables and figures

```bash
python -m privtdd reproduce --mode figures
```

This verifies the immutable `results/published/paper-v1` snapshot, recomputes
statistics and intervals, rebuilds E1 and E4 tables, the four-network E2 curves,
the Higgs/Orkut E3 figure, supplementary component comparisons, and checks the
E3 explanatory statistics. It writes to `results/reproduced/paper-v1` and also
builds an extended supplementary LaTeX source with E4 and implementation details.
It does not execute the private estimators or require the real graph files.

To also compile the extended supplement, install a LaTeX distribution with
`latexmk` and the packages used by the provided source:

```bash
python -m privtdd reproduce --mode figures --compile-supplement
```

The bundled PDF is [paper/supplement/supplementary.pdf](paper/supplement/supplementary.pdf).
The historical E1–E3 content is retained separately. Chinese editing comments
have been translated or removed in the public source; active TeX content is
unchanged and original source hashes are retained in provenance. Numeric source
tables are provided; raw historical identities are never rewritten as new runs.

## Run the small end-to-end check

```bash
python -m privtdd reproduce --mode smoke
```

This executes PrivTDD on a synthetic graph, the main component and boundary
variants, the E4 sampling smoke cases, and fresh-result aggregation/plots.
It uses reduced integration settings and produces clearly marked smoke outputs.
It is not evidence for the paper's real-network numerical results.

## Prepare the real datasets and run fresh experiments

```bash
python -m privtdd data prepare
python -m privtdd doctor --verify-data
python -m privtdd reproduce --mode full --plan
```

Data are fetched from the registered official sources, checked, and converted
using fixed node-ID mappings. If a source is unavailable, place its documented
raw file in `datasets/raw` and retry. See [datasets/README.md](datasets/README.md).
Large raw graphs, processed arrays and working caches are not Git assets.

Prepared-data bundles can also be hosted on Google Drive. See
[cloud-data instructions](docs/cloud_data.md) for per-dataset bundles,
SHA-256 verification, installation, and the official-source fallback. The
cloud manifest includes the public file links in the
[prepared-data Drive folder](https://drive.google.com/drive/folders/1xuxagBfpbpQYqq7mOQy0kVtiY15FG-zW?usp=sharing).
The downloader verifies each archive against the recorded SHA-256 before
installing it. Per-file validation status is recorded in the manifest.

LDPGraph synthesis reads a separately supplied upstream checkout. This project
does not redistribute that repository. Read [THIRD_PARTY.md](THIRD_PARTY.md) for
the exact commit, required source hashes, source conditions and adaptations.
After preparing that external source, the complete fresh pipeline is:

```bash
python -m privtdd reproduce --mode full --external-code-root /path/to/LDPGraph
```

It executes the current paper's E1–E3 and supplementary ablations, then the forty
E4 sampled releases using the new E1 controls. It records failures and generates
summaries from the new runs, not from the frozen paper snapshot. Run the same
command to resume; add `--retry-failed` to retry failed tasks. A nonzero exit
status reports incomplete or failed work. No heavy real-network experiment is
run by the install or figure commands.

For an explicit subset without LDPGraph:

```bash
python -m privtdd reproduce --mode full --suite E1 --methods PrivTDD
```

Use `--data-root`, `--output`, `--workers` and `--memory-gb` to set storage and
resource limits. The memory limit is a guard, not a measured requirement. E4
alone needs the matching E1 run root selected by `--output`. When installed
outside an editable checkout, use `python -m privtdd --repo-root /path/to/PrivTDD ...`.

## Read the implementation

The API exports `PrivTDDConfig`, `estimate_tdd`, and
`estimate_tdd_from_reports`. Its generic numerical defaults are separate from
the explicit paper preset.

| Paper phase | Package modules |
|---|---|
| Private reports and base distributions | `reporting`, `degree`, `base_distribution` |
| Complementary structural references | `references` |
| Report-driven correction | `correction` |
| Posterior aggregation and final release | `posterior`, `release` |

See [method and naming](docs/method.md), [reproduction details](docs/reproduction.md),
and [release inventory](docs/release.md). `provenance/module_mapping.json` records
the old-to-new source mapping; old method IDs in frozen records are historical
labels, not alternative public APIs.

## License and citation

The authors' code is licensed under [MIT](LICENSE). Third-party notices and
dataset conditions remain separate; the MIT license does not relicense the
upstream LDPGraph code or real datasets. Citation metadata are in
[CITATION.cff](CITATION.cff). This artifact does not claim paper acceptance or a
conference artifact badge.
