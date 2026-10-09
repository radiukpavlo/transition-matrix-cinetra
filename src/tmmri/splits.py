from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def acdc_patient_split(
    inventory: pd.DataFrame,
    seed: int = 20260611,
    validation_fraction_of_training: float = 0.15,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    train_inv = inventory[inventory["source_split"] == "training"].copy()
    test_inv = inventory[inventory["source_split"] == "testing"].copy()
    rng = np.random.default_rng(seed)

    val_patients: set[str] = set()
    for _, group_df in train_inv.groupby("disease_label", dropna=False):
        patient_ids = sorted(group_df["patient_id"].astype(str).unique())
        if not patient_ids:
            continue
        n_val = max(1, int(round(len(patient_ids) * validation_fraction_of_training)))
        chosen = list(rng.choice(patient_ids, size=min(n_val, len(patient_ids)), replace=False))
        val_patients.update(chosen)

    for row in train_inv.itertuples(index=False):
        patient_id = str(row.patient_id)
        rows.append(
            {
                "patient_id": patient_id,
                "dataset": "ACDC",
                "split": "val" if patient_id in val_patients else "train",
                "label": row.disease_label,
                "seed": seed,
                "split_version": "acdc_patient_split_v2",
            }
        )
    for row in test_inv.itertuples(index=False):
        rows.append(
            {
                "patient_id": str(row.patient_id),
                "dataset": "ACDC",
                "split": "test",
                "label": row.disease_label,
                "seed": seed,
                "split_version": "acdc_patient_split_v2",
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "patient_id"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- protocol 6.0.0 fold rules
# Decision D15 / D17; every salt, fold count and expected count is read from configs/splits.yaml.  The legacy
# functions acdc_patient_split and select_mms_100 above and below are unchanged (governed and legacy flows).


def fold_digest(salt: str, key: str) -> str:
    """``SHA256(salt + key)`` hex digest (the v6 fold hash)."""

    import hashlib

    return hashlib.sha256(f"{salt}{key}".encode("utf-8")).hexdigest()


def hash_rank_folds(frame: pd.DataFrame, key: str, strata: str, salt: str, n_folds: int) -> pd.Series:
    """Within each stratum sort by ``SHA256(salt + key)`` ascending (ties by key); fold = (0-based rank mod n) + 1."""

    if frame[key].duplicated().any():
        raise ValueError(f"duplicate {key} values in the fold frame")
    folds = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    for _, group in frame.groupby(strata, dropna=False, sort=True):
        order = sorted(group.index, key=lambda i: (fold_digest(salt, str(frame.at[i, key])), str(frame.at[i, key])))
        for rank, index in enumerate(order):
            folds.at[index] = rank % int(n_folds) + 1
    return folds


def acdc_outer_folds_v6(cohort: pd.DataFrame, splits_cfg: dict) -> pd.DataFrame:
    """Outer folds of ``acdc_dev`` (official training patients) per ``splits.yaml`` ``acdc.acdc_dev.outer_folds``.

    ``cohort`` needs ``patient_id``, ``acdc_class`` and ``partition``; returns ``sample_id, patient_id, acdc_class,
    outer_fold`` sorted by ``sample_id``.  Folds are assigned on the *eligible* patients only (``eligible_v6`` when the
    column exists), so a patient excluded by a declared duplicate image (``configs/cohorts.yaml``
    ``duplicate_image_pairs``) never takes a fold rank.
    """

    cfg = splits_cfg["acdc"]["acdc_dev"]["outer_folds"]
    keep = cohort["partition"] == "acdc_dev"
    if "eligible_v6" in cohort.columns:
        keep &= cohort["eligible_v6"].astype(bool)
    dev = cohort[keep].copy()
    dev["outer_fold"] = hash_rank_folds(dev, str(cfg["key"]), "acdc_class", str(cfg["hash_salt"]), int(cfg["n_folds"]))
    return dev[["sample_id", "patient_id", "acdc_class", "outer_fold"]].sort_values("sample_id").reset_index(drop=True)


def acdc_inner_folds_v6(outer: pd.DataFrame, outer_fold: int, splits_cfg: dict) -> pd.DataFrame:
    """Inner folds inside the training set of ``outer_fold``: salt ``acdc-inner-fold-v6|`` + str(f) + ``|``."""

    cfg = splits_cfg["acdc"]["acdc_dev"]["inner_folds"]
    train = outer[outer["outer_fold"] != int(outer_fold)].copy()
    salt = f"{cfg['hash_salt']}{int(outer_fold)}|"
    train["inner_fold"] = hash_rank_folds(train, "patient_id", "acdc_class", salt, int(cfg["n_folds"]))
    train["outer_fold_heldout"] = int(outer_fold)
    return train[["sample_id", "patient_id", "acdc_class", "outer_fold_heldout", "inner_fold"]].sort_values("sample_id").reset_index(drop=True)


def acdc_all_inner_folds_v6(outer: pd.DataFrame, splits_cfg: dict) -> pd.DataFrame:
    n = int(splits_cfg["acdc"]["acdc_dev"]["outer_folds"]["n_folds"])
    return pd.concat([acdc_inner_folds_v6(outer, f, splits_cfg) for f in range(1, n + 1)], ignore_index=True)


def mms_refit_folds_v6(cohort: pd.DataFrame, splits_cfg: dict) -> pd.DataFrame:
    """Refit-arm inner folds of ``mms_refit_train`` stratified by vendor (``splits.yaml`` ``refit_arm.inner_folds``)."""

    cfg = splits_cfg["refit_arm"]["inner_folds"]
    train = cohort[cohort["refit_partition"] == "mms_refit_train"].copy()
    train["refit_fold"] = hash_rank_folds(train, "case_id", str(cfg["strata"]), str(cfg["hash_salt"]), int(cfg["n_folds"]))
    return train[["sample_id", "case_id", "vendor", "refit_fold"]].sort_values("sample_id").reset_index(drop=True)


def fold_balance(frame: pd.DataFrame, fold_col: str, strata: str) -> pd.DataFrame:
    """Counts per fold and stratum (for the expected-count checks of ``splits.yaml``)."""

    return frame.groupby([fold_col, strata]).size().unstack(fill_value=0).sort_index()


def build_split_manifest_v6(acdc_cohort: pd.DataFrame | None, mms_cohort: pd.DataFrame | None, splits_cfg: dict) -> pd.DataFrame:
    """One row per case: partition, ACDC outer fold, M&Ms strata membership and refit-arm partition/fold."""

    frames = []
    if acdc_cohort is not None:
        outer = acdc_outer_folds_v6(acdc_cohort, splits_cfg)
        acdc = acdc_cohort.merge(outer[["sample_id", "outer_fold"]], on="sample_id", how="left")
        acdc = acdc.assign(partition=acdc["partition"].where(acdc["eligible_v6"], "excluded"), mms100_member=False,
                           never_analysed=False, refit_partition=None, refit_fold=pd.NA, stratum=acdc["acdc_class"])
        frames.append(acdc)
    if mms_cohort is not None:
        refit = mms_refit_folds_v6(mms_cohort, splits_cfg)
        mms = mms_cohort.merge(refit[["sample_id", "refit_fold"]], on="sample_id", how="left")
        mms = mms.assign(partition=np.where(mms["mms_external"], "mms_external", "excluded"), outer_fold=pd.NA,
                         stratum=mms["vendor"])
        frames.append(mms)
    columns = ["sample_id", "dataset", "case_id", "patient_id", "release_split", "partition", "stratum", "outer_fold",
               "mms100_member", "never_analysed", "refit_partition", "refit_fold", "eligible_v6", "exclusion_reason",
               "cluster_id", "duplicate_pair_id", "duplicate_excluded_from", "duplicate_reason_code"]
    out = pd.concat([f.reindex(columns=columns) for f in frames], ignore_index=True)
    out["outer_fold"] = out["outer_fold"].astype("Int64")
    out["refit_fold"] = out["refit_fold"].astype("Int64")
    out["cluster_id"] = out["cluster_id"].where(out["cluster_id"].notna(), out["sample_id"])
    out["split_rule_version"] = "acdc-fold-v6"
    return out.sort_values("sample_id").reset_index(drop=True)


# ----------------------------------------------------------------------------- declared duplicate images (v6)
# PI decision of September 28, 2026 (protocol/amendments.md; SPECIFICATION.md 4.5, 12.2 and 13.1).  The pairs and
# their actions are read from configs/cohorts.yaml ``duplicate_image_pairs``; validate-cohort detects duplicates from
# the decoded staged image voxels and fails on any pair that is not declared here.

DUPLICATE_ACTIONS = ("exclude", "exclude_and_cluster", "cluster")
PHASE_TOKENS = {"ed_image": "ED", "es_image": "ES"}


def duplicate_image_pairs(cohorts_cfg: dict) -> list[dict]:
    """The declared pairs (``configs/cohorts.yaml`` ``duplicate_image_pairs.pairs``), validated."""

    pairs = list((cohorts_cfg.get("duplicate_image_pairs") or {}).get("pairs") or [])
    seen: set[str] = set()
    for p in pairs:
        ids = [str(i) for i in p.get("ids", [])]
        if len(ids) != 2 or ids[0] == ids[1]:
            raise ValueError(f"duplicate_image_pairs {p.get('id')!r}: ids must name two different cases")
        if str(p.get("action")) not in DUPLICATE_ACTIONS:
            raise ValueError(f"duplicate_image_pairs {p.get('id')!r}: action must be one of {DUPLICATE_ACTIONS}")
        exclude = p.get("exclude")
        if ("exclude" in str(p["action"])) != bool(exclude):
            raise ValueError(f"duplicate_image_pairs {p.get('id')!r}: an exclude action needs exactly one exclude entry")
        if exclude and (str(exclude.get("case")) not in ids or not exclude.get("from") or not p.get("reason_code")):
            raise ValueError(f"duplicate_image_pairs {p.get('id')!r}: exclude needs case (a member), from (a partition) and reason_code")
        if str(p.get("id")) in seen:
            raise ValueError(f"duplicate_image_pairs: id {p.get('id')!r} declared twice")
        seen.add(str(p.get("id")))
    return pairs


def pair_cluster_id(dataset: str, ids: list[str]) -> str:
    """``<DATASET>:<id1>+<id2>`` (ids sorted): the shared bootstrap cluster of a declared pair."""

    return f"{dataset}:" + "+".join(sorted(str(i) for i in ids))


def apply_duplicate_declarations(cohort: pd.DataFrame | None, dataset: str, cohorts_cfg: dict) -> pd.DataFrame | None:
    """Apply the declared actions to one cohort table (idempotent).

    Adds ``cluster_id`` (``sample_id`` for a singleton; :func:`pair_cluster_id` for both members of a declared pair),
    ``duplicate_pair_id``, ``duplicate_excluded_from`` and ``duplicate_reason_code``.  An exclusion from the primary
    partition of the case (``acdc_dev``, ``acdc_test`` or ``mms_external``) makes the case ineligible
    (``eligible_v6`` False, ``exclusion_reason`` = the reason code); an exclusion from ``mms_refit_eval`` or
    ``mms_refit_train`` clears ``refit_partition`` only, so the case stays in ``mms_external``.  Pairs whose members
    are not in the table (for example synthetic fixtures) are ignored.
    """

    if cohort is None:
        return None
    out = cohort.copy()
    out["cluster_id"] = out["sample_id"].astype(str)
    for column in ("duplicate_pair_id", "duplicate_excluded_from", "duplicate_reason_code"):
        out[column] = pd.Series([None] * len(out), index=out.index, dtype=object)
    key = str(dataset).lower()
    by_case = {str(c): i for i, c in zip(out.index, out["case_id"])}
    for pair in duplicate_image_pairs(cohorts_cfg):
        if str(pair.get("cohort", "")).lower() != key:
            continue
        ids = [str(i) for i in pair["ids"]]
        rows = [by_case[i] for i in ids if i in by_case]
        if not rows:
            continue
        out.loc[rows, "duplicate_pair_id"] = str(pair["id"])
        out.loc[rows, "cluster_id"] = pair_cluster_id(str(out.at[rows[0], "dataset"]), ids)
        exclude = pair.get("exclude")
        if not exclude or str(exclude["case"]) not in by_case:
            continue
        row, target, code = by_case[str(exclude["case"])], str(exclude["from"]), str(pair["reason_code"])
        out.at[row, "duplicate_excluded_from"] = target
        out.at[row, "duplicate_reason_code"] = code
        if target in {"mms_refit_eval", "mms_refit_train"}:
            if "refit_partition" in out.columns and out.at[row, "refit_partition"] == target:
                out.at[row, "refit_partition"] = None
        elif target in {"acdc_dev", "acdc_test", "mms_external"}:
            out.at[row, "eligible_v6"] = False
            out.at[row, "exclusion_reason"] = code
            for column in ("mms_external", "never_analysed"):
                if column in out.columns:
                    out.at[row, column] = False
            if "refit_partition" in out.columns:
                out.at[row, "refit_partition"] = None
        else:
            raise ValueError(f"duplicate_image_pairs {pair['id']!r}: unknown partition {target!r}")
    return out


def detect_image_duplicates(records: list[dict]) -> dict[tuple[str, str], list[str]]:
    """Pairs of staged cases whose decoded ED or ES image voxels are identical.

    ``records`` are staging-manifest case records (``sample_id`` and ``staged.<name>.voxel_sha256``).  Returns
    ``{(sample_a, sample_b): frames}`` with ``sample_a < sample_b`` and frames such as ``["ED", "ES"]`` (a same-phase
    match) or ``"ED/ES"`` (the ED image of ``sample_a`` equals the ES image of ``sample_b``).
    """

    by_digest: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rec in records:
        for name, entry in (rec.get("staged") or {}).items():
            if name in PHASE_TOKENS and entry.get("voxel_sha256"):
                by_digest[str(entry["voxel_sha256"])].append((str(rec["sample_id"]), PHASE_TOKENS[name]))
    found: dict[tuple[str, str], set[str]] = defaultdict(set)
    for members in by_digest.values():
        for i, (sa, pa) in enumerate(members):
            for sb, pb in members[i + 1:]:
                if sa == sb:
                    continue
                (s1, p1), (s2, p2) = sorted([(sa, pa), (sb, pb)])
                found[(s1, s2)].add(p1 if p1 == p2 else f"{p1}/{p2}")
    return {pair: sorted(frames) for pair, frames in sorted(found.items())}


def declared_duplicate_findings(detected: dict[tuple[str, str], list[str]], staged_ids: set[str], cohorts_cfg: dict) -> dict[str, list]:
    """Compare detected duplicates with the declarations.

    ``undeclared``: detected pairs that are not declared (validate-cohort fails on any).  ``frame_mismatch``: declared
    pairs whose two members are staged but whose detected frames differ from ``matched_frames`` (including a declared
    pair that is no longer detected).  ``declared``: one row per declared pair with its detected frames.
    """

    datasets = {"acdc": "ACDC", "mms": "MMS"}
    declared: dict[tuple[str, str], dict] = {}
    for pair in duplicate_image_pairs(cohorts_cfg):
        ds = datasets.get(str(pair.get("cohort", "")).lower(), str(pair.get("cohort", "")).upper())
        declared[tuple(sorted(f"{ds}:{i}" for i in pair["ids"]))] = pair
    undeclared = [[a, b, frames] for (a, b), frames in detected.items() if (a, b) not in declared]
    mismatch, rows = [], []
    for (a, b), pair in declared.items():
        frames = detected.get((a, b))
        both = a in staged_ids and b in staged_ids
        expected = sorted(str(f) for f in pair.get("matched_frames", []))
        if both and frames != expected:
            mismatch.append([a, b, {"declared": expected, "detected": frames or []}])
        rows.append({"pair_id": str(pair["id"]), "samples": [a, b], "declared_frames": expected, "detected_frames": frames,
                     "both_staged": both, "action": str(pair["action"]), "reason_code": pair.get("reason_code"),
                     "exclude": pair.get("exclude")})
    return {"undeclared": undeclared, "frame_mismatch": mismatch, "declared": rows}


def duplicate_action_violations(tables: dict[str, pd.DataFrame | None], split_manifest: pd.DataFrame | None, cohorts_cfg: dict) -> list:
    """Declared actions that are not reflected in the cohort tables and the split manifest (expected: none)."""

    datasets = {"acdc": "ACDC", "mms": "MMS"}
    problems: list = []
    for pair in duplicate_image_pairs(cohorts_cfg):
        ds = datasets.get(str(pair.get("cohort", "")).lower())
        table = tables.get(ds) if ds else None
        if table is None:
            continue
        sub = table[table["case_id"].astype(str).isin([str(i) for i in pair["ids"]])]
        if len(sub) < 2:
            continue
        cluster = pair_cluster_id(ds, [str(i) for i in pair["ids"]])
        if set(sub["cluster_id"].astype(str)) != {cluster}:
            problems.append([pair["id"], "cluster_id_not_shared"])
        exclude = pair.get("exclude")
        if not exclude:
            continue
        sid = f"{ds}:{exclude['case']}"
        target = str(exclude["from"])
        if split_manifest is not None:
            row = split_manifest[split_manifest["sample_id"] == sid]
            if row.empty:
                problems.append([pair["id"], f"{sid}_missing_from_split_manifest"])
                continue
            r = row.iloc[0]
            in_target = (r["refit_partition"] if target in {"mms_refit_eval", "mms_refit_train"} else r["partition"]) == target
            if in_target or r["duplicate_reason_code"] != pair["reason_code"]:
                problems.append([pair["id"], f"{sid}_still_in_{target}"])
            if target == "acdc_dev" and pd.notna(r["outer_fold"]):
                problems.append([pair["id"], f"{sid}_has_outer_fold"])
    return problems


def cluster_counts(frame: pd.DataFrame, mask: pd.Series | np.ndarray) -> dict[str, int]:
    """Cases and independent bootstrap clusters (distinct ``cluster_id``) of the rows selected by ``mask``."""

    sel = frame[np.asarray(mask, dtype=bool)]
    return {"cases": int(len(sel)), "clusters": int(sel["cluster_id"].astype(str).nunique())}


def partition_cluster_report(split_manifest: pd.DataFrame, mms_cohort: pd.DataFrame | None, strata: list[str],
                             refit_groups: dict | None = None) -> dict[str, object]:
    """Cases and clusters per partition, per ACDC class, per external stratum and per ``acdc_dev`` outer fold."""

    sm = split_manifest
    report: dict[str, object] = {"partitions": {}, "acdc_classes": {}, "mms_external_strata": {}, "mms_refit_eval_strata": {}}
    parts = {
        "acdc_dev": sm["partition"] == "acdc_dev",
        "acdc_test": sm["partition"] == "acdc_test",
        "mms_external": sm["partition"] == "mms_external",
        "mms_external_never_analysed": (sm["partition"] == "mms_external") & sm["never_analysed"].fillna(False).astype(bool),
        "mms100_member": (sm["partition"] == "mms_external") & sm["mms100_member"].fillna(False).astype(bool),
        "mms_refit_train": sm["refit_partition"] == "mms_refit_train",
        "mms_refit_eval": sm["refit_partition"] == "mms_refit_eval",
    }
    for name, mask in parts.items():
        report["partitions"][name] = cluster_counts(sm, mask)
    for part in ("acdc_dev", "acdc_test"):
        sub = sm[sm["partition"] == part]
        report["acdc_classes"][part] = {str(c): cluster_counts(sub, sub["stratum"] == c) for c in sorted(sub["stratum"].dropna().astype(str).unique())}
    dev = sm[(sm["partition"] == "acdc_dev") & sm["outer_fold"].notna()]
    report["acdc_dev_outer_folds"] = {
        str(int(f)): {**cluster_counts(dev, dev["outer_fold"] == f),
                      "by_class": {str(k): int(v) for k, v in dev.loc[dev["outer_fold"] == f, "stratum"].value_counts().sort_index().items()}}
        for f in sorted(dev["outer_fold"].dropna().unique())}
    if mms_cohort is not None and "cluster_id" in mms_cohort.columns:
        ext = mms_cohort[mms_cohort["mms_external"].astype(bool)]
        for var in list(strata) + ["never_analysed"]:
            if var not in ext.columns:
                continue
            values = ext[var].astype(object).where(ext[var].notna(), "<missing>").astype(str)
            report["mms_external_strata"][var] = {v: cluster_counts(ext, values == v) for v in sorted(values.unique())}
        ev = mms_cohort[mms_cohort["refit_partition"] == "mms_refit_eval"]
        vendor = ev["vendor"].astype(str)
        report["mms_refit_eval_strata"]["vendor"] = {v: cluster_counts(ev, vendor == v) for v in sorted(vendor.unique())}
        if refit_groups:
            for group in ("seen_vendors", "unseen_vendors"):
                members = {str(v) for v in refit_groups.get(group, [])}
                report["mms_refit_eval_strata"][group] = cluster_counts(ev, vendor.isin(members))
    return report


def _stable_case_hash(case_id: str) -> str:
    import hashlib

    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()


def select_mms_100(inventory: pd.DataFrame, selected_count: int = 100, seed: int = 20260611) -> pd.DataFrame:
    eligible = inventory[inventory["eligible_for_mms_100"]].copy()
    eligible = eligible.drop_duplicates("patient_id").copy()
    if len(eligible) < selected_count:
        out = eligible.copy()
        out["selected_rank"] = pd.NA
        out["selection_seed"] = seed
        out["selection_reason"] = "not_enough_eligible_cases"
        return out

    eligible["stable_case_hash"] = eligible["case_id"].astype(str).map(_stable_case_hash)
    sort_cols = ["vendor", "center", "disease_label", "stable_case_hash", "case_id"]
    eligible = eligible.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    groups: dict[tuple[object, object, object], list[int]] = defaultdict(list)
    for idx, row in eligible.iterrows():
        groups[(row.get("vendor"), row.get("center"), row.get("disease_label"))].append(idx)

    group_keys = sorted(groups.keys(), key=lambda k: tuple("" if x is None or pd.isna(x) else str(x) for x in k))
    cursors = {key: 0 for key in group_keys}
    selected_indices: list[int] = []
    while len(selected_indices) < selected_count:
        made_progress = False
        for key in group_keys:
            cursor = cursors[key]
            items = groups[key]
            if cursor >= len(items):
                continue
            selected_indices.append(items[cursor])
            cursors[key] += 1
            made_progress = True
            if len(selected_indices) == selected_count:
                break
        if not made_progress:
            break

    selected = eligible.loc[selected_indices].copy()
    selected = selected.reset_index(drop=True)
    selected["selected_rank"] = np.arange(1, len(selected) + 1)
    selected["selection_seed"] = seed
    selected["selection_reason"] = "round_robin_vendor_center_pathology_stable_hash"
    selected["eligibility_status"] = "eligible"
    return selected.drop(columns=["stable_case_hash"], errors="ignore")
