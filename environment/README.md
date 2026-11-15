# Environment provenance

The manuscript states Windows 11 Pro, Python 3.12.9, Intel Core i7-12700 and
32 GiB memory. Historical E4 records also contain Python 3.12.9. A complete
historical package lock was not recovered; inconsistent old requirements files
were not relabeled as an authoritative lock.

The migration and local validation environment was Python 3.12.7 (Anaconda),
NumPy 1.26.4, SciPy 1.13.1, Numba 0.60.0, llvmlite 0.43.0, threadpoolctl 3.5.0,
psutil 5.9.0, filelock 3.13.1, NetworkX 3.3, pandas 2.2.2 and matplotlib 3.9.2.
`constraints-windows-py312.txt` pins these tested dependency versions. It does
not force Python 3.12.7 rather than 3.12.9; use a Python 3.12 interpreter.

Cross-platform installation and full real-network execution have not been
claimed from the Windows tests alone. The CI workflow checks lightweight
installation, tests and frozen-statistics reproduction on Linux when run.
Full experiments are not CI jobs. Actual time and peak memory must be read
from run records, rather than inferred from graph file sizes or worker limits.

To compare a new environment, run the core tests, smoke pipeline and frozen
statistics checks first. Fresh runs get their own code/environment records and
output roots. Do not overwrite `results/published/paper-v1`.
