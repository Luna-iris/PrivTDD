"""Degree optimization phases for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from privtdd.degree.reduced import refine_reduced

from privtdd.degree.working_set import refine_working_set



# [PrivTDD] S2 degree prior; numerical solver (not an additional statistical model); Screen the active set first, retaining its result for reduced Newton if needed; iteration limits are finite, with no repeated verification loop.
def refine_two_phase(caps,public_cap,eps_degree,initial,*,tolerance=1e-7,
                     screening_iterations=200,refinement_iterations=2000,checkpoint=None):
    last={}
    # [PrivTDD] S2 degree prior; numerical solver (not an additional statistical model); Record the current first-phase iteration and forward the optional progress callback.
    def first_checkpoint(p,iteration,gap,value):
        last.update(prior=p.copy(),iteration=iteration,gap=gap,nll=value)
        if checkpoint is not None:checkpoint(p,iteration,gap,value)
    try:
        prior,audit=refine_working_set(caps,public_cap,eps_degree,initial,tolerance=tolerance,
            max_iterations=screening_iterations,checkpoint=first_checkpoint)
        return prior,dict(audit,numerical_phases=1)
    except ArithmeticError as error:
        if not str(error).startswith('Working-set Newton did not meet unchanged precision:'):raise
        if not last or last['iteration']!=screening_iterations:
            raise ArithmeticError('Complete phase-one checkpoint required') from error
        failure=str(error)
    # [PrivTDD] S2 degree prior; numerical solver (not an additional statistical model); Continue second-phase iteration numbering after the first phase and forward the callback.
    def second_checkpoint(p,iteration,gap,value):
        if checkpoint is not None:checkpoint(p,screening_iterations+iteration,gap,value)
    prior,audit=refine_reduced(caps,public_cap,eps_degree,last['prior'],tolerance=tolerance,
        max_iterations=refinement_iterations,checkpoint=second_checkpoint)
    return prior,dict(audit,numerical_phases=2,screening_phase_terminal=dict(iteration=last['iteration'],
        gap=last['gap'],nll=last['nll'],active_states=int((last['prior']>0).sum())),
        screening_phase_failure=failure,repeated_screening_in_final_phase=False,
        statistical_objective_unchanged=True)
