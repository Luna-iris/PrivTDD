"""S4: Beta and concentric-circle Geo conditional references from the same private reports."""
import json
import numpy as np
from privtdd.references.context import structural_context, prepare_compositions
from privtdd.references.independent_edge import generate as generate_endpoints
from privtdd.references.clustering import generate as generate_geometry
from privtdd.io.storage import save_sources, load_sources, json_write


# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); [Core S4 / structural model] Return (Beta, Geo) R(T|d), preserving the fixed degree marginal.
def build_structural_joint(caps, prior, eps_degree, private_mean, directory, config):
    """[Core S4 / structural model] Return (Beta, Geo) R(T|d), preserving the fixed degree marginal.
    
    caps/prior/eps_degree: Private degree-posterior inputs from the same experiment; private_mean: mean of released noisy_T.
    directory: Cache with a bound identity; config.geometry_draws/replicates: numerical integration budget.
    Beta counts by type pairs; Geo calibrates kappa from the private mean, without using true triangle statistics.
    """
    path = directory/'sources.npz'
    if path.exists():
        return load_sources(path), json.loads((directory/'sources.json').read_text(encoding='utf-8'))
    context, _ = structural_context(caps, prior, eps_degree)
    packet, _ = prepare_compositions(context, draws=config.geometry_draws, replicates=config.geometry_replicates,
                                     reuse_root_kernels=config.compiled_structure)
    # endpoint_mean[0] reduces the original endpoint generator to the independent-edge endpoint; its independent RNG streams are unchanged.
    from privtdd.references.calibration import calibrate_projected_geometry
    calibration, _ = calibrate_projected_geometry(packet,caps,prior,eps_degree,private_mean)
    if config.structural_lazy_endpoints and calibration['kappa'] != 1.:
        from privtdd.references.independent_edge import generate_beta
        beta, beta_draws = generate_beta(packet, compiled=config.compiled_structure)
        # kappa=0 uses Beta; interior kappa does not read endpoint samples; kappa=1 uses the full endpoint path.
        endpoints = np.column_stack((beta_draws, np.zeros_like(beta_draws))) if calibration['kappa'] == 0. else None
    else:
        laws, draws, _ = generate_endpoints(packet, float(packet['endpoint_means'][0]), compiled=config.compiled_structure)
        beta, endpoints = laws[0], draws[:, :2]
    geometry, _, stats = generate_geometry(packet, private_mean, endpoint_draws=endpoints, calibration=calibration,
                                           compiled=config.compiled_structure)
    stats['skipped_unused_hard_endpoint'] = bool(calibration is not None and calibration['kappa'] != 1.)
    stats['compiled_structure'] = config.compiled_structure
    sources = (beta, geometry)
    json_write(directory/'sources.json', stats)
    save_sources(path, sources)
    return sources, stats
