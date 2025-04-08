"""Utility file containing miscellaneous common functions"""

from logging import getLogger

logger = getLogger(__name__)


def check_results(results, SolutionStatus, TerminationCondition):
    """Check results for termination condition and solution status

    Parameters
    ----------
    results : str
        Results from pyomo
    SolutionStatus : str
        Solution Status from pyomo
    TerminationCondition : str
        Termination Condition from pyomo

    Returns
    -------
    results
    """
    return (
        (results is None)
        or (len(results.solution) == 0)
        or (results.solution(0).status == SolutionStatus.infeasible)
        or (results.solver.termination_condition == TerminationCondition.infeasible)
        or (results.solver.termination_condition == TerminationCondition.unbounded)
    )
