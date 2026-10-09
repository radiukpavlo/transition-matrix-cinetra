"""Descriptive demographic strata of the external evaluation (SPECIFICATION.md 12.3; Appendix A Q28)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tmmri.config import load_study
from tmmri.stages import descriptive_strata


def test_age_bands_and_sex_labels_follow_the_configured_edges():
    study = load_study()
    assert study.bootstrap["subgroup_reporting_thresholds"]["age_band_edges_years"] == [40, 65]
    ids = [f"MMS:C{i}" for i in range(7)]
    cohort = pd.DataFrame({"age_years": [15.0, 39.9, 40.0, 64.0, 64.9, 65.0, np.nan],
                           "sex": ["F", "M", "M", "F", None, "M", "F"]}, index=ids)
    out = descriptive_strata(study, cohort, ids)
    assert set(out) == {"sex", "age_band"}
    assert list(out["age_band"]) == ["<40", "<40", "40-64", "40-64", "40-64", ">=65", "<missing>"]
    assert list(out["sex"]) == ["F", "M", "M", "F", "<missing>", "M", "F"]
