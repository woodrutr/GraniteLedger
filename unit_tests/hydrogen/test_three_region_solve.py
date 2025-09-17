import pytest

pyo = pytest.importorskip(
    'pyomo.environ', reason='Pyomo is required for hydrogen model tests'
)

from pathlib import Path

from definitions import PROJECT_ROOT
from src.models.hydrogen.model import actions

import logging

logger = logging.getLogger(__name__)


@pytest.mark.skip('not ready: need stable costs in dataset')
def test_three_reg_solve():
    """test that h2 run provides expected objective in 3 region example"""
    # run the model
    data_path = Path(PROJECT_ROOT / 'input/hydrogen/three_region')
    grid_data = actions.load_data(data_path)
    grid = actions.build_grid(grid_data=grid_data)
    model = actions.build_model(grid=grid)
    sol = actions.solve_it(model)

    # pull the objective value out for examination...
    obj_val = pyo.value(model.cost_expression)
    assert obj_val == pytest.approx(2671.56), "objective value doesn't match"
