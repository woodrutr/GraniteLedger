import pandas as pd

from gui.outputs_visualization import region_selection_options
from gui.region_metadata import DEFAULT_REGION_METADATA


def test_region_selection_options_uses_metadata_labels():
    df = pd.DataFrame(
        {
            "region_canonical": [1, 2, pd.NA, 2],
            "region_label": ["custom", "labels", pd.NA, "labels"],
        }
    )

    options = region_selection_options(df)

    expected = [
        (DEFAULT_REGION_METADATA[1].label, 1),
        (DEFAULT_REGION_METADATA[2].label, 2),
    ]
    assert options == expected


def test_region_selection_options_falls_back_to_metadata_for_default_entries():
    df = pd.DataFrame(
        {
            "region_canonical": ["default", "default"],
            "region_label": ["default", "default"],
        }
    )

    options = region_selection_options(df)

    assert len(options) == len(DEFAULT_REGION_METADATA)
    assert {value for _, value in options} == set(DEFAULT_REGION_METADATA)
    assert all(label != "default" for label, _ in options)
