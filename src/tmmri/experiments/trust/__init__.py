"""Trust analyses S1-S4 of the protocol-v6.0.1 amendment (docs/MANUSCRIPT_PROTOCOL_V2_2026-10-02.md, Section E).

Pure functions on arrays: selective prediction (:mod:`.selective`), label-free OOD scores (:mod:`.ood`), label-free
harmonization of A (:mod:`.harmonize`) and calibration transfer (:mod:`.calibration`).  The stages that fit them on
``acdc_dev`` out-of-fold outputs and evaluate them once per partition live in :mod:`tmmri.experiments.l57_stages`.
Research use only; estimation, not diagnosis.
"""

from __future__ import annotations
