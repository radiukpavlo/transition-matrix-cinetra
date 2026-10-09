"""Registry contract of the typed reference table B (schema ``B_v6.0``; SPECIFICATION.md 6.0.0, Section 6).

:func:`validate_typed_b` checks a typed B table against ``registry/semantic_registry.yaml`` and raises
:class:`ContractViolation` with the full list of violations:

* required metadata columns present and non-null; ``schema_version == B_v6.0``;
* every registry feature has its value column and the ``__state``/``__evidence`` companions;
* no column outside the registry (metadata, QC, features and companions) unless ``allow_extra_columns``;
* dtypes per the registry (``float64``, ``int64``, ``bool``, ``string``; all-null columns pass);
* enums of metadata columns; ``sample_id == f"{dataset_source}:{case_id}"``, unique and sorted ascending;
* states in the registry vocabulary; values null unless the state is ``OBSERVED``, and non-null when it is;
* OBSERVED values finite and inside the registry domain (``[low, high]``, null bound = unbounded; ``null_only`` =
  never observed) on every row: an out-of-domain value on a critical-QC row is ``QC_BLOCKED`` by
  :func:`tmmri.semantics.features.case_semantics_v6`, so no exemption exists; evidence tokens registered and allowed
  for the cohort (``evidence_sources``; tiers joined by ``;``, ``|``, ``,`` or ``+``);
* ``qc_failure_reason`` tokens registered in ``configs/cohorts.yaml`` ``qc_severity`` (fail closed) when the
  cohort config is given.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

TYPED_SCHEMA = "B_v6.0"
EVIDENCE_SEPARATORS = (";", "|", ",", "+")   # METHODS joins tiers with "+"
_COHORT_BY_DATASET = {"ACDC": "acdc", "MMS": "mms", "MMS2": "mms"}   # MMS2: evidence profile of M&Ms-1 (cohorts.yaml mms2)


class ContractViolation(ValueError):
    """Raised with ``violations`` (a list of human-readable strings)."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = list(violations)
        preview = "; ".join(self.violations[:10])
        more = f" (+{len(self.violations) - 10} more)" if len(self.violations) > 10 else ""
        super().__init__(f"{len(self.violations)} contract violation(s): {preview}{more}")


def registry_features(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Registry features by name, in registry order."""

    return {str(f["name"]): dict(f) for f in registry.get("features", [])}


def typed_columns(registry: Mapping[str, Any]) -> list[str]:
    """The full typed column list: metadata, QC, then each feature with its two companions."""

    companions = list(registry.get("typed_companions", ["__state", "__evidence"]))
    columns = list(registry.get("metadata_columns", {})) + list(registry.get("qc_columns", {}))
    for name in registry_features(registry):
        columns.extend([name, *(name + c for c in companions)])
    return columns


def _dtype_ok(series: pd.Series, dtype: str) -> bool:
    if series.isna().all():
        return True
    if dtype == "float64":
        return pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series)
    if dtype == "int64":
        return pd.api.types.is_integer_dtype(series)
    if dtype == "bool":
        return pd.api.types.is_bool_dtype(series)
    if dtype == "string":
        values = series.dropna()
        return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series) and values.map(lambda v: isinstance(v, str)).all()
    return True


def _domain(spec: Mapping[str, Any]) -> tuple[float | None, float | None] | str | None:
    domain = spec.get("domain")
    if domain is None:
        return None
    if isinstance(domain, str):
        return domain
    low, high = domain
    return (None if low is None else float(low), None if high is None else float(high))


def _tokens(value: object) -> list[str]:
    text = str(value)
    for sep in EVIDENCE_SEPARATORS[1:]:
        text = text.replace(sep, EVIDENCE_SEPARATORS[0])
    return [t.strip() for t in text.split(EVIDENCE_SEPARATORS[0]) if t.strip()]


def _registered_qc_reasons(cohorts_cfg: Mapping[str, Any]) -> set[str]:
    severity = cohorts_cfg.get("qc_severity", {})
    return {str(r) for level in ("critical", "high", "moderate", "informational") for r in severity.get(level, [])}


def _examples(index: Iterable[Any], frame: pd.DataFrame, limit: int = 3) -> str:
    ids = list(frame.loc[list(index), "sample_id"].astype(str)[:limit]) if "sample_id" in frame.columns else list(index)[:limit]
    return ", ".join(ids)


def validate_typed_b(
    frame: pd.DataFrame,
    registry: Mapping[str, Any],
    *,
    cohorts_cfg: Mapping[str, Any] | None = None,
    allow_extra_columns: bool = False,
    require_all_features: bool = True,
) -> dict[str, Any]:
    """Validate ``frame`` (see the module docstring); returns a summary dict or raises :class:`ContractViolation`."""

    violations: list[str] = []
    schema = str(registry.get("typed_schema_version", TYPED_SCHEMA))
    states = set(registry.get("states", []))
    tiers = set(registry.get("evidence_tiers", {}))
    companions = list(registry.get("typed_companions", ["__state", "__evidence"]))
    state_suffix, evidence_suffix = companions[0], companions[1]
    metadata = dict(registry.get("metadata_columns", {}))
    qc = dict(registry.get("qc_columns", {}))
    features = registry_features(registry)
    known = set(typed_columns(registry))

    for name, spec in metadata.items():
        if name not in frame.columns:
            if spec.get("required"):
                violations.append(f"missing required metadata column {name}")
            continue
        if spec.get("required") and frame[name].isna().any():
            violations.append(f"required column {name} has nulls ({_examples(frame.index[frame[name].isna()], frame)})")
    for name, spec in {**metadata, **qc}.items():
        if name in frame.columns and not _dtype_ok(frame[name], str(spec.get("dtype", ""))):
            violations.append(f"column {name}: dtype {frame[name].dtype} is not registry dtype {spec.get('dtype')}")
        enum = spec.get("enum")
        if name in frame.columns and enum:
            bad = frame[name].dropna()
            bad = bad[~bad.isin(list(enum))]
            if len(bad):
                violations.append(f"column {name}: values outside enum {sorted(set(map(str, bad)))[:5]}")
    if not allow_extra_columns:
        extra = [c for c in frame.columns if c not in known]
        if extra:
            violations.append(f"columns not in the registry: {extra[:10]}")

    if "schema_version" in frame.columns:
        wrong = frame["schema_version"].astype(str) != schema
        if wrong.any():
            violations.append(f"schema_version must be {schema} ({int(wrong.sum())} rows)")
    if "sample_id" in frame.columns:
        ids = frame["sample_id"].astype(str)
        if ids.duplicated().any():
            violations.append(f"sample_id not unique ({sorted(ids[ids.duplicated()])[:3]})")
        if list(ids) != sorted(ids):
            violations.append("rows are not sorted by sample_id")
        if {"dataset_source", "case_id"}.issubset(frame.columns):
            expected = frame["dataset_source"].astype(str) + ":" + frame["case_id"].astype(str)
            mismatch = ids != expected
            if mismatch.any():
                violations.append(f"sample_id != dataset_source:case_id ({_examples(frame.index[mismatch], frame)})")

    cohort = frame["dataset_source"].map(_COHORT_BY_DATASET) if "dataset_source" in frame.columns else pd.Series(None, index=frame.index)
    # No critical-QC exemption: the builder marks an out-of-domain value on a critical-QC row QC_BLOCKED (D21), so an
    # OBSERVED value outside the registry domain is always a contract violation.
    for name, spec in features.items():
        state_col, evidence_col = name + state_suffix, name + evidence_suffix
        present = [c for c in (name, state_col, evidence_col) if c in frame.columns]
        if len(present) < 3:
            if require_all_features or present:
                missing = sorted({name, state_col, evidence_col} - set(present))
                violations.append(f"feature {name}: missing columns {missing}")
            continue
        value, state, evidence = frame[name], frame[state_col], frame[evidence_col]
        if not _dtype_ok(value, str(spec.get("dtype", "float64"))):
            violations.append(f"feature {name}: dtype {value.dtype} is not {spec.get('dtype')}")
        bad_state = state.isna() | ~state.isin(list(states))
        if bad_state.any():
            violations.append(f"feature {name}: state outside {sorted(states)} ({_examples(frame.index[bad_state], frame)})")
        observed = state.eq("OBSERVED")
        leaked = value.notna() & ~observed
        if leaked.any():
            violations.append(f"feature {name}: non-null value with state != OBSERVED ({_examples(frame.index[leaked], frame)})")
        hollow = value.isna() & observed
        if hollow.any():
            violations.append(f"feature {name}: OBSERVED with a null value ({_examples(frame.index[hollow], frame)})")
        domain = _domain(spec)
        numeric = pd.to_numeric(value.where(observed), errors="coerce")
        if domain == "null_only":
            if observed.any():
                violations.append(f"feature {name}: reserved (null_only) but OBSERVED ({_examples(frame.index[observed], frame)})")
        elif isinstance(domain, tuple):
            low, high = domain
            outside = pd.Series(False, index=frame.index)
            if low is not None:
                outside |= numeric < low
            if high is not None:
                outside |= numeric > high
            if (~np.isfinite(numeric.fillna(0.0).to_numpy(dtype=float))).any():
                outside |= ~np.isfinite(numeric.fillna(0.0).to_numpy(dtype=float))
            if outside.any():
                violations.append(f"feature {name}: OBSERVED value outside domain {list(domain)} ({_examples(frame.index[outside], frame)})")
        sources = spec.get("evidence_sources", {})
        for index in frame.index[observed]:
            raw = evidence.at[index]
            if raw is None or (isinstance(raw, float) and np.isnan(raw)) or not str(raw).strip():
                violations.append(f"feature {name}: OBSERVED without evidence ({_examples([index], frame)})")
                break
            tokens = _tokens(raw)
            unknown = [t for t in tokens if t not in tiers]
            allowed = sources.get(cohort.at[index]) if isinstance(sources, Mapping) else None
            if unknown:
                violations.append(f"feature {name}: unregistered evidence {unknown} ({_examples([index], frame)})")
                break
            if isinstance(allowed, list) and any(t not in allowed for t in tokens):
                violations.append(f"feature {name}: evidence {tokens} not in evidence_sources for {cohort.at[index]} ({_examples([index], frame)})")
                break
            if allowed == "UNSUPPORTED":
                violations.append(f"feature {name}: OBSERVED although the cohort evidence is UNSUPPORTED ({_examples([index], frame)})")
                break

    for name, spec in qc.items():
        domain = _domain(spec)
        if name in frame.columns and isinstance(domain, tuple):
            numeric = pd.to_numeric(frame[name], errors="coerce")
            low, high = domain
            outside = (numeric < low if low is not None else False) | (numeric > high if high is not None else False)
            if np.any(outside):
                violations.append(f"qc column {name}: value outside domain {list(domain)}")
    if cohorts_cfg is not None and "qc_failure_reason" in frame.columns:
        registered = _registered_qc_reasons(cohorts_cfg)
        tokens = {t.split(":", 1)[0].strip() for v in frame["qc_failure_reason"].dropna() for t in str(v).split(";") if t.strip()}
        unknown = sorted(tokens - registered)
        if unknown:
            violations.append(f"qc_failure_reason tokens not registered in configs/cohorts.yaml (QC_UNREGISTERED_REASON): {unknown[:10]}")

    if violations:
        raise ContractViolation(violations)
    return {"schema": schema, "rows": int(len(frame)), "features": len(features), "columns": int(frame.shape[1])}
