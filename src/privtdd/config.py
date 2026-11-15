"""Fixed method design and adjustable implementation/numerical controls; no dataset-specific parameters."""
from dataclasses import dataclass
import math

# [Method design / reporting mechanism] One undirected edge affects both endpoint degrees/releases and one shared RR response.
DEGREE_FRACTION = 0.15
RR_FRACTION = 0.35
RELEASE_FRACTION = 0.175
CAP_OFFSET = 0
# Numerical coverage radius is independent of the reporting offset L.
DEGREE_GRID_COVERAGE_RADIUS = 75
# [Method design / F1 and joint family] Retain the version used in the completed experiments.
CLOSURE_CELLS = 65
POSITIVE_QUANTILES = (0.90, 0.95)
JOINT_COORDINATES = (0, 1, 3, 4, 5, 6, 7)
DEGREE_TOLERANCE = 1e-7
CLOSURE_TOLERANCE = 1e-7
JOINT_TOLERANCE = 1e-6
JOINT_MAX_ITERATIONS = 1000


@dataclass(frozen=True)
class PrivTDDConfig:
    """[Implementation/numerical configuration] Defaults use the integrated path; legacy is the F1 numerical path used for Higgs.
    
    workers: Threads for S1 neighbor positions and postprocessing cap/degree tasks; samples and degree support are unchanged.
    S1 respects the Numba runtime thread limit and restores the previous thread setting on return.
    blas_threads: BLAS threads per process; limits nested parallelism.
    posterior_backend: integrated uses the current common-kernel integration; legacy integrates the original components in blocks.
    geometry_draws: Nominal structural integration budget per replicate, allocated by pi (rounding up can increase the actual total).
    geometry_replicates: Number of fixed structural random streams; not independent privacy experiment seeds.
    numerical_W1_limit: Numerical W1 target for F1 integration at each degree; not a statistical risk bound.
    compression_W1_budget: Global transport error budget for F1 compression.
    report_block_size: Number of root nodes per block in the original-graph reporting simulator.
    degree_grid_size: 0 retains the full integer-degree solver; >=68 enables the candidate degree grid approximation.
    report_bin_width: 0 disables report compression; >0 bins by Laplace noise scale within each cap.
    degree_grid_tolerance: Optimization gap budget per report for negative log likelihood on the approximate grid; does not apply to the original exact path.
    triangle_bin_width: 0 uses the original W1 compression; >0 discretizes T at the mechanism noise scale, falling back to W1 compression if the grid is too fine.
    structural_lazy_endpoints: Omit unused hard endpoints while retaining the original structural random streams and sampling budget.
    batch_small_groups: Batch common-kernel forward/adjoint computations across small cap groups using a uniform workload threshold; do not merge scales.
    batch_f1: Batch F1 by degree and directly integrate small groups of reports with the same scale; retain each report's original noise scale.
    compiled_structure: Cache fixed kernels within an experiment, batch Beta and compile deterministic Geo counts; NumPy retains the original sampling.
    safeguard: Apply the capacity-feasible S7 transport safeguard after the single S6 posterior update; enabled by default on every numerical path.
    safeguard_tau: Public preset tau in transport radius tau*D, default 1; not selected using truth or an individual result.
    Grid and bin widths define uniform numerical approximations, not thread/cache optimizations; they do not guarantee improvement over a baseline.
    """
    workers: int = 8
    blas_threads: int = 2
    posterior_backend: str = 'integrated'
    geometry_draws: int = 65536
    geometry_replicates: int = 4
    numerical_W1_limit: float = 1e-6
    compression_W1_budget: float = 2.0
    report_block_size: int = 256
    degree_grid_size: int = 0
    report_bin_width: float = 0.0
    degree_grid_tolerance: float = 1e-5
    # [Numerical approximation] After conversion from T to Q, the bin width is at most this fraction of the smallest effective Laplace scale; 0 uses the original W1 compression.
    triangle_bin_width: float = 0.0
    # [Equivalent implementation optimization] Generate hard endpoint samples only when Geo requires them.
    structural_lazy_endpoints: bool = False
    # [Common-kernel implementation] Select the original sparse integration or direct batched kernel by workload; do not bin caps/scales.
    batch_small_groups: bool = False
    # [Common-kernel implementation] Batch F1 by latent degree and directly integrate small scale groups; components, bin widths and stopping requirements are unchanged.
    batch_f1: bool = False
    # [Same-model implementation] Compile structural sample loops and reuse fixed root-type kernels and tilted probabilities; do not reduce samples.
    compiled_structure: bool = False
    # S7: final postprocessing, common to exact/binned/fast and both F1 backends.
    safeguard: bool = True
    safeguard_tau: float = 1.0

    # [PrivTDD] S0 configuration; core method or probability/structure primitive (not necessarily a separate contribution); [Input validation] Reject invalid control values; do not read data or select a model.
    def __post_init__(self):
        """[Input validation] Reject invalid control values; do not read data or select a model."""
        if not isinstance(self.safeguard, bool):
            raise ValueError('safeguard must be boolean')
        if not math.isfinite(self.safeguard_tau) or self.safeguard_tau <= 0:
            raise ValueError('safeguard_tau must be finite and positive')
        for name in ('workers', 'blas_threads', 'geometry_draws', 'geometry_replicates', 'report_block_size'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.posterior_backend not in ('integrated', 'legacy'):
            raise ValueError('posterior_backend must be integrated or legacy')
        if isinstance(self.degree_grid_size, bool) or not isinstance(self.degree_grid_size, int) or (self.degree_grid_size != 0 and self.degree_grid_size < 68):
            raise ValueError('degree_grid_size must be 0 or an integer >=68')
        if not math.isfinite(self.report_bin_width) or self.report_bin_width < 0:
            raise ValueError('report_bin_width must be finite and nonnegative')
        if not math.isfinite(self.triangle_bin_width) or self.triangle_bin_width < 0:
            raise ValueError('triangle_bin_width must be finite and nonnegative')
        if not isinstance(self.structural_lazy_endpoints, bool):
            raise ValueError('structural_lazy_endpoints must be boolean')
        if not isinstance(self.batch_small_groups, bool):
            raise ValueError('batch_small_groups must be boolean')
        if not isinstance(self.batch_f1, bool) or not isinstance(self.compiled_structure, bool):
            raise ValueError('batch_f1 and compiled_structure must be boolean')
        for name in ('numerical_W1_limit', 'compression_W1_budget', 'degree_grid_tolerance'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
