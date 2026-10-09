"""Protocol-v6.0.1 M&Ms-2 ingest and lock on synthetic fixtures (never cohort data): split manifest, extraction list,
release-CSV contract, geometry rule, label identity, the lock (exit 3 before the tag), sealed rows never written and
the decoded-voxel duplicate rule."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fixtures_mms2 as f2
import fixtures_v6 as fx
from tmmri import cli, stages
from tmmri.config import load_study
from tmmri.ingest import mms2
from tmmri.ingest.labels import DATASET_LABEL_MAPPINGS
from tmmri.ingest.sources import SourceError
from tmmri.ingest.staging import load_staged_case, stage_dataset
from tmmri.mms2_stages import validate_cohort_mms2
from tmmri.runs import RunRoot
from tmmri.stages import ProtocolFreezeError, StageContext

EXTERNAL = mms2.external_ids()
SEALED = mms2.sealed_ids()


# ----------------------------------------------------------------------------- split manifest


def test_split_manifest_is_reproducible_from_the_rule(tmp_path):
    committed = mms2.load_split_manifest()                      # verify=True: equals the rule
    rebuilt = mms2.build_split_manifest()
    assert committed.equals(rebuilt)
    assert mms2.build_split_manifest(subjects=list(reversed(mms2.SUBJECT_IDS))).equals(rebuilt)    # independent of input order
    assert set(mms2.external_ids(mms2.build_split_manifest(salt="other-salt|"))) != set(mms2.external_ids(rebuilt))   # salt-dependent
    assert len(EXTERNAL) == len(SEALED) == 180 and not set(EXTERNAL) & set(SEALED)
    assert set(EXTERNAL) | set(SEALED) == set(mms2.SUBJECT_IDS) == {f"{i:03d}" for i in range(1, 361)}
    import hashlib

    order = sorted(mms2.SUBJECT_IDS, key=lambda s: (hashlib.sha256(f"mms2-split-v6.0.1|{s}".encode()).hexdigest(), s))
    assert sorted(order[:180]) == EXTERNAL
    receipt = mms2.load_receipt()
    assert receipt["split_manifest"]["sha256"] == mms2.lf_sha256(mms2.SPLIT_MANIFEST)
    assert receipt["archive"]["sha256"] == "934076fe388f0747c80b7f0fdf5ebc57cd480ab737be8228639480e61515362b"
    assert receipt["archive"]["bytes"] == 9824397247
    edited = committed.copy()
    edited.loc[0, "partition"], edited.loc[359, "partition"] = edited.loc[359, "partition"], edited.loc[0, "partition"]
    edited.to_csv(tmp_path / "split.csv", index=False)
    with pytest.raises(SourceError, match="does not reproduce"):
        mms2.load_split_manifest(tmp_path / "split.csv")


# ----------------------------------------------------------------------------- archive listing and extraction list


def _listing() -> list[mms2.ArchiveEntry]:
    """The release layout observed before the tag (member names only): 3,964 entries."""

    entries = [mms2.ArchiveEntry("MnM2/", 0, False), mms2.ArchiveEntry("MnM2/dataset/", 0, False),
               mms2.ArchiveEntry("MnM2/dataset_information.csv", 900, True), mms2.ArchiveEntry("MnM2/readme.txt", 40, True)]
    for sid in mms2.SUBJECT_IDS:
        entries.append(mms2.ArchiveEntry(f"MnM2/dataset/{sid}/", 0, False))
        for suffix in ("SA_ED", "SA_ED_gt", "SA_ES", "SA_ES_gt", "SA_CINE", "LA_ED", "LA_ED_gt", "LA_ES", "LA_ES_gt", "LA_CINE"):
            entries.append(mms2.ArchiveEntry(f"MnM2/dataset/{sid}/{sid}_{suffix}.nii.gz", 1000 + int(sid), True))
    return entries


def test_extraction_list_is_exactly_the_720_external_short_axis_files():
    names = mms2.extraction_names(EXTERNAL)
    assert len(names) == 720 == len(set(names))
    assert all(n.split("_", 1)[0] in set(EXTERNAL) for n in names)
    assert {n.split("_", 1)[1] for n in names} == {"SA_ED.nii.gz", "SA_ED_gt.nii.gz", "SA_ES.nii.gz", "SA_ES_gt.nii.gz"}
    selection = mms2.select_members(_listing(), EXTERNAL, info_csv_name="dataset_information.csv")
    members = selection["members"]
    assert len(members) == 720 and sorted(m["member"].rsplit("/", 1)[1] for m in members) == names
    assert not any("LA_" in m["member"] or "CINE" in m["member"] for m in members)
    assert {m["case_id"] for m in members} == set(EXTERNAL) and not {m["case_id"] for m in members} & set(SEALED)
    assert selection["counts"]["short_axis_ed_es_files"] == 1440 and selection["info_csv"].member == "MnM2/dataset_information.csv"
    layout = mms2.layout_report(selection, _cfg()["inputs"]["archive_layout"])
    assert layout["listing_entries_observed"] == 3964 and layout["listing_entries_match"] and layout["info_csv_matches"]
    assert layout["members_matching_template"] == 720 and layout["members_differing"] == []
    moved = [mms2.ArchiveEntry(e.member.replace("MnM2/dataset/", "MnM2/other/"), e.size, e.is_file) for e in _listing()]
    other = mms2.layout_report(mms2.select_members(moved, EXTERNAL, info_csv_name="dataset_information.csv"), _cfg()["inputs"]["archive_layout"])
    assert other["members_matching_template"] == 0                  # informational: recorded in the receipt, never blocking
    missing = [e for e in _listing() if not e.member.endswith(f"{EXTERNAL[0]}_SA_ES_gt.nii.gz")]
    with pytest.raises(mms2.HumanGateError, match="layout"):
        mms2.select_members(missing, EXTERNAL, info_csv_name="dataset_information.csv")
    doubled = _listing() + [mms2.ArchiveEntry(f"copy/{EXTERNAL[1]}_SA_ED.nii.gz", 5, True)]
    with pytest.raises(mms2.HumanGateError):
        mms2.select_members(doubled, EXTERNAL, info_csv_name="dataset_information.csv")
    no_sealed = [e for e in _listing() if f"/{SEALED[0]}/" not in e.member]     # every subject 001-360 must be listed (names only)
    with pytest.raises(mms2.HumanGateError, match="not exactly 001-360"):
        mms2.select_members(no_sealed, EXTERNAL, info_csv_name="dataset_information.csv")
    # A sealed subject must also carry each of its four short-axis ED/ES members exactly once (SPECIFICATION 4.6).
    sealed_gap = [e for e in _listing() if not e.member.endswith(f"{SEALED[0]}_SA_ES_gt.nii.gz")]
    with pytest.raises(mms2.HumanGateError, match=f"{SEALED[0]}_SA_ES_gt.nii.gz: 0 archive file entries"):
        mms2.select_members(sealed_gap, EXTERNAL, info_csv_name="dataset_information.csv")
    sealed_twice = _listing() + [mms2.ArchiveEntry(f"MnM2/extra/{SEALED[1]}_SA_ED.nii.gz", 7, True)]
    with pytest.raises(mms2.HumanGateError, match=f"{SEALED[1]}_SA_ED.nii.gz: 2 archive file entries"):
        mms2.select_members(sealed_twice, EXTERNAL, info_csv_name="dataset_information.csv")


def test_release_csv_is_read_into_memory_only_and_never_falls_back_to_a_file(tmp_path, monkeypatch):
    calls = []

    def fake_tar(tar, args, *, timeout=None):
        calls.append(list(args))
        return mms2.subprocess.CompletedProcess(args, 1, b"", b"simulated -xO failure")

    monkeypatch.setattr(mms2, "_tar", fake_tar)
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(mms2.HumanGateError, match="into memory"):
        mms2.read_member_bytes(tmp_path / "MnM2.rar", "MnM2/dataset_information.csv", "tar")
    assert calls == [["-xOf", str(tmp_path / "MnM2.rar"), "MnM2/dataset_information.csv"]]   # no -xf to a temporary directory
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    monkeypatch.setattr(mms2, "_tar", lambda tar, args, timeout=None: mms2.subprocess.CompletedProcess(args, 0, b"", b""))
    with pytest.raises(mms2.HumanGateError):                       # empty output fails closed as well
        mms2.read_member_bytes(tmp_path / "MnM2.rar", "MnM2/dataset_information.csv", "tar")


def test_parse_listing_bsdtar_and_gnu_formats():
    text = ("drwxrwxrwx  0 0      0           0 ??? 02 21:51 MnM2/\n"
            "-rw-rw-rw-  0 0      0        1234 Oct  2  2021 MnM2/dataset/001/001_SA_ED.nii.gz\n"
            "-rw-r--r-- user/group 99 2026-10-02 21:51 MnM2/dataset/002/002_SA_ES_gt.nii.gz\n")
    entries = mms2.parse_listing(text)
    assert [(e.path, e.size, e.is_file) for e in entries] == [("MnM2/", 0, False), ("MnM2/dataset/001/001_SA_ED.nii.gz", 1234, True),
                                                              ("MnM2/dataset/002/002_SA_ES_gt.nii.gz", 99, True)]
    with pytest.raises(mms2.HumanGateError, match="unparsed"):
        mms2.parse_listing("garbage line\n")


# ----------------------------------------------------------------------------- release CSV contract


def _cfg():
    return load_study().cohorts["mms2"]


def test_release_csv_contract_drops_sealed_rows_before_parsing_values():
    rows = [f2.info_row(sid, i) for i, sid in enumerate(mms2.SUBJECT_IDS)]
    ext, report = mms2.parse_release_csv(f2.info_csv_bytes(rows), _cfg(), EXTERNAL, expect_all_subjects=True)
    assert list(ext["__case_id"]) == EXTERNAL and report["rows_dropped_not_external"] == 180
    copy = mms2.external_csv_bytes(ext).decode("utf-8")
    assert copy.splitlines()[0] == ",".join(f2.INFO_COLUMNS)
    assert {mms2.normalize_subject_code(line.split(",")[0]) for line in copy.splitlines()[1:]} == set(EXTERNAL)
    # An undeclared value in a sealed row is never parsed; in an external row it fails closed.
    sealed_bad = [dict(r, DISEASE="XYZ") if mms2.normalize_subject_code(r["SUBJECT_CODE"]) in SEALED else r for r in rows]
    mms2.parse_release_csv(f2.info_csv_bytes(sealed_bad), _cfg(), EXTERNAL, expect_all_subjects=True)
    external_bad = [dict(r, DISEASE="XYZ") if mms2.normalize_subject_code(r["SUBJECT_CODE"]) == EXTERNAL[0] else r for r in rows]
    with pytest.raises(mms2.HumanGateError, match="DISEASE"):
        mms2.parse_release_csv(f2.info_csv_bytes(external_bad), _cfg(), EXTERNAL, expect_all_subjects=True)
    for columns in (f2.INFO_COLUMNS + ["CENTRE"], f2.INFO_COLUMNS[:-1], ["", *f2.INFO_COLUMNS]):
        extra = [dict(r, CENTRE="1", **{"": "0"}) for r in rows]
        with pytest.raises(mms2.HumanGateError, match="columns"):
            mms2.parse_release_csv(f2.info_csv_bytes(extra, columns), _cfg(), EXTERNAL, expect_all_subjects=True)
    with pytest.raises(mms2.HumanGateError, match="repeats"):
        mms2.parse_release_csv(f2.info_csv_bytes(rows + rows[:1]), _cfg(), EXTERNAL, expect_all_subjects=True)
    with pytest.raises(mms2.HumanGateError, match="differ from 001-360"):
        mms2.parse_release_csv(f2.info_csv_bytes(rows[:-1]), _cfg(), EXTERNAL, expect_all_subjects=True)
    vendor = [dict(r, VENDOR="Philips Medical Systems") for r in rows]
    ext, _ = mms2.parse_release_csv(f2.info_csv_bytes(vendor), _cfg(), EXTERNAL, expect_all_subjects=True)
    table = mms2.info_table(ext, _cfg())
    assert set(table["vendor"]) == {"Philips"} and set(table["vendor_release"]) == {"Philips Medical Systems"}
    # The header read before the tag: no SEX or AGE column; either one fails closed as an unknown column.
    for extra_column in ("SEX", "AGE"):
        with pytest.raises(mms2.HumanGateError, match=extra_column):
            mms2.parse_release_csv(f2.info_csv_bytes([dict(r, **{extra_column: "1"}) for r in rows], f2.INFO_COLUMNS + [extra_column]),
                                   _cfg(), EXTERNAL, expect_all_subjects=True)
    # LF line endings parse like the release's CRLF.
    lf, _ = mms2.parse_release_csv(f2.info_csv_bytes(rows, line_end="\n"), _cfg(), EXTERNAL, expect_all_subjects=True)
    assert lf.equals(mms2.parse_release_csv(f2.info_csv_bytes(rows), _cfg(), EXTERNAL, expect_all_subjects=True)[0])


def test_release_format_observed_before_the_tag_parses_rv_spelling_and_blank_rows():
    """Decision D-2 (3 October 2026): the release spells dilated RV "RV", uses these vendor strings and carries
    separator-only blank rows; a subject row with an empty declared value still fails closed."""

    observed = ["NOR", "LV", "HCM", "ARR", "FALL", "CIA", "TRI", "RV"]
    vendors = ["SIEMENS", "GE MEDICAL SYSTEMS", "Philips Medical Systems"]
    rows = [dict(f2.info_row(sid, i), DISEASE=observed[i % 8], VENDOR=vendors[i % 3], FIELD=["1.5", "3"][i % 2])
            for i, sid in enumerate(mms2.SUBJECT_IDS)]
    data = f2.info_csv_bytes(rows) + b",,,,\r\n,,,,\r\n"
    ext, _ = mms2.parse_release_csv(data, _cfg(), EXTERNAL, expect_all_subjects=True)
    table = mms2.info_table(ext, _cfg())
    assert len(table) == 180
    assert set(table["pathology"]) <= {"NOR", "LV", "HCM", "ARR", "FALL", "CIA", "TRI", "DRV"} and "RV" not in set(table["pathology"])
    assert set(table["vendor"]) <= {"Siemens", "GE", "Philips"}
    empty = [dict(r, DISEASE="") if mms2.normalize_subject_code(r["SUBJECT_CODE"]) == EXTERNAL[0] else r for r in rows]
    with pytest.raises(mms2.HumanGateError):
        mms2.info_table(mms2.parse_release_csv(f2.info_csv_bytes(empty), _cfg(), EXTERNAL, expect_all_subjects=True)[0], _cfg())


def test_vendor_scanner_field_are_data_driven_and_disease_synonyms_fail_closed_only_outside_the_table():
    rows = [f2.info_row(sid, i) for i, sid in enumerate(mms2.SUBJECT_IDS)]
    odd = {"SIEMENS": "Siemens", "siemens healthineers": "Siemens", "GE MEDICAL SYSTEMS": "GE", "General Electric": "GE",
           "Philips Ingenia": "Philips", "Canon": "Canon", "Toshiba Vantage": "Toshiba Vantage", "Siemens/Philips": "Siemens/Philips"}
    for raw, normalized in odd.items():
        assert mms2.normalize_vendor(raw) == normalized
    assert mms2.normalize_vendor("  ") is None
    values = list(odd)
    unusual = [dict(r, VENDOR=values[i % len(values)], SCANNER=f" scanner {i} ", FIELD=["1.5T", "3", "", "unknown"][i % 4])
               for i, r in enumerate(rows)]
    ext, _ = mms2.parse_release_csv(f2.info_csv_bytes(unusual), _cfg(), EXTERNAL, expect_all_subjects=True)   # never fails closed
    table = mms2.info_table(ext, _cfg()).set_index("case_id")
    first = ext.iloc[0]
    assert table.at[first["__case_id"], "field_strength"] == (first["FIELD"].strip() or None)
    assert table.at[first["__case_id"], "vendor_release"] == first["VENDOR"]
    assert table.at[first["__case_id"], "vendor"] == mms2.normalize_vendor(first["VENDOR"])
    synonyms = {"Normal": "NOR", " dilated LEFT ventricle ": "LV", "Hypertrophic cardiomyopathy": "HCM", "Congenital arrhythmogenesis": "ARR",
                "ARRHYTHMOGENIC": "ARR", "Tetralogy of  Fallot": "FALL", "interatrial communication": "CIA", "Tricuspid regurgitation": "TRI",
                "Dilated right ventricle": "DRV", "nor": "NOR"}
    spelled = [dict(r, DISEASE=list(synonyms)[i % len(synonyms)]) for i, r in enumerate(rows)]
    ext, _ = mms2.parse_release_csv(f2.info_csv_bytes(spelled), _cfg(), EXTERNAL, expect_all_subjects=True)
    got = mms2.info_table(ext, _cfg()).set_index("case_id")["pathology"]
    checked = 0
    for r in spelled:
        code = mms2.normalize_subject_code(r["SUBJECT_CODE"])
        if code in got.index:
            assert got[code] == synonyms[r["DISEASE"]]
            checked += 1
    assert checked == 180
    bad = [dict(r, DISEASE="Dilated cardiomyopathy") if mms2.normalize_subject_code(r["SUBJECT_CODE"]) == EXTERNAL[0] else r for r in rows]
    with pytest.raises(mms2.HumanGateError, match="DISEASE"):
        mms2.parse_release_csv(f2.info_csv_bytes(bad), _cfg(), EXTERNAL, expect_all_subjects=True)


def test_case_ids_are_prefixed_downstream_and_the_split_keeps_the_raw_code():
    assert mms2.case_key("001") == "MMS2_001" and mms2.subject_of("MMS2_360") == "360"
    for bad in ("001", "MMS2_000", "MMS2_361", "mms2_001"):
        with pytest.raises(SourceError):
            mms2.subject_of(bad)
    assert mms2.load_split_manifest()["subject_id"].str.fullmatch(r"\d{3}").all()


def test_crosswalk_same_definition_and_non_equivalent_strata():
    cfg = _cfg()
    assert {k for k, v in cfg["crosswalk"].items() if v["compatibility"] == stages.SAME_DEFINITION} == {"NOR", "HCM"}
    assert cfg["crosswalk"]["LV"] == {**cfg["crosswalk"]["LV"], "acdc": "DCM", "compatibility": "HARMONIZABLE_NOT_EXACT", "pathology_group": "DLV"}
    rv = {k for k, v in cfg["crosswalk"].items() if v["pathology_group"] == "RV_FAMILY"}
    assert rv == {"ARR", "FALL", "CIA", "TRI", "DRV"} and {cfg["crosswalk"][k]["acdc"] for k in rv} == {"RV"}
    assert "MINF" not in {v["acdc"] for v in cfg["crosswalk"].values()}
    assert set(cfg["crosswalk"]) == set(cfg["info_csv"]["columns"]["DISEASE"]["values"])


# ----------------------------------------------------------------------------- staging: geometry and labels


def _stage(root: Path, out: Path):
    cohort = mms2.mms2_cohort_table(root, cohort_cfg=_cfg())
    return stage_dataset("mms2", root, out, n_jobs=1, cohort=cohort, manifests_dir=out.parent / "manifests"), cohort


def test_geometry_rule_and_label_identity(tmp_path):
    subjects = EXTERNAL[:4]
    base = f2.case_arrays(subjects[3])
    root = f2.make_mms2_root(tmp_path / "mms2", subjects[:3], geometry={subjects[0]: "voxel", subjects[1]: "unit", subjects[2]: "inconsistent"})
    # a fourth case whose mask header disagrees with its image header
    rels = f2.write_case(root, subjects[3], base, mask_affine=np.diag([1.25, 1.25, 6.0, 1.0]))
    manifest = mms2.load_file_manifest(root)
    cases = {sid: {r.role: r.relpath for r in manifest.itertuples() if r.case_id == mms2.case_key(sid)} for sid in subjects[:3]}
    f2.finish_root(root, {**cases, subjects[3]: rels}, [f2.info_row(s, i) for i, s in enumerate(subjects)])
    report, cohort = _stage(root, tmp_path / "staged")
    failures = {f["case_id"]: f for f in report.failures}
    assert set(failures) == {mms2.case_key(subjects[2]), mms2.case_key(subjects[3])}
    assert {f["reason_code"] for f in failures.values()} == {mms2.GEOMETRY_REASON}
    voxel = load_staged_case("MMS2", mms2.case_key(subjects[0]), tmp_path / "staged")
    unit = load_staged_case("MMS2", mms2.case_key(subjects[1]), tmp_path / "staged")
    assert voxel.geometry_method == "nifti_affine_projection" and voxel.spacing_mm == (1.25, 1.25, 8.0)
    assert unit.geometry_method == "nifti_header_pixdim_unscaled_affine" and unit.spacing_mm == pytest.approx(f2.UNIT_ZOOMS)
    assert unit.meta["geometry_check"]["affine_scaling"]["ed_mask"] == "unit_scaled"
    assert voxel.ed_frame_index is None and voxel.meta["phase_source"] == "file_name"
    arrays = f2.case_arrays(subjects[0])
    for phase in ("ed", "es"):           # identity label map: staged masks equal the released 1 LV, 2 MYO, 3 RV
        assert np.array_equal(getattr(voxel, f"{phase}_mask"), arrays[f"{phase}_mask"].astype(np.int16))
        assert np.array_equal(getattr(voxel, f"{phase}_image"), arrays[f"{phase}_image"])
    assert DATASET_LABEL_MAPPINGS["MMS2"] == load_study().label_map("mms2") == {0: 0, 1: 1, 2: 2, 3: 3}
    assert set(cohort["crosswalk_compatibility"].dropna()) <= {stages.SAME_DEFINITION, "HARMONIZABLE_NOT_EXACT"}


def test_file_manifest_refuses_a_sealed_subject(tmp_path):
    root = f2.make_mms2_root(tmp_path / "mms2", [EXTERNAL[0]])
    frame = pd.read_csv(root / mms2.FILE_MANIFEST, dtype=str, keep_default_na=False)
    frame.loc[0, "case_id"] = mms2.case_key(SEALED[0])
    frame.to_csv(root / mms2.FILE_MANIFEST, index=False)
    with pytest.raises(SourceError, match="outside mms2_external"):
        mms2.load_file_manifest(root)
    frame.loc[0, "case_id"] = EXTERNAL[0]                          # a raw numeric code is never a case id
    frame.to_csv(root / mms2.FILE_MANIFEST, index=False)
    with pytest.raises(SourceError, match="not an M&Ms-2 case id"):
        mms2.load_file_manifest(root)


# ----------------------------------------------------------------------------- extraction and the lock


def _context(tmp_path, *, allow=False, sources=None) -> StageContext:
    study = load_study()
    return StageContext(study=study, run=None, data_root=tmp_path / "data", n_jobs=1, allow_unfrozen_synthetic=allow, sources=dict(sources or {}))


def test_extraction_refuses_before_the_tag_then_verifies_the_hash(tmp_path, monkeypatch):
    tar = f2.find_tar()
    if tar is None:
        pytest.skip("no tar executable")
    archive = f2.make_mms2_archive(tmp_path / "archive")
    dest = tmp_path / "raw" / "mms2"
    monkeypatch.setattr(stages, "protocol_frozen", lambda *a, **k: False)
    ctx = _context(tmp_path)
    read = []
    monkeypatch.setattr(mms2, "verify_archive", lambda *a, **k: read.append(1))
    with pytest.raises(ProtocolFreezeError, match="protocol-v6.0.1"):     # the lock comes first: no archive byte is read
        mms2.extract_external(archive, dest, cohort_cfg=_cfg(), tar=tar,
                              require_unlocked=lambda: stages.require_mms2_unlocked(ctx, "extract-mms2", synthetic=True))
    assert read == [] and not dest.exists()
    monkeypatch.undo()
    descriptor = archive.parent / mms2.SYNTHETIC_ARCHIVE_DESCRIPTOR
    good = descriptor.read_text(encoding="utf-8")
    descriptor.write_text(json.dumps({**json.loads(good), "sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(SourceError, match="SHA-256 mismatch"):         # after the lock, a different archive extracts nothing
        mms2.extract_external(archive, dest, cohort_cfg=_cfg(), tar=tar, require_unlocked=lambda: "protocol_frozen")
    assert not dest.exists()
    descriptor.write_text(good, encoding="utf-8")
    monkeypatch.setattr(stages, "protocol_frozen", lambda *a, **k: False)
    code = None
    try:
        cli.main(["extract-mms2", "--archive", str(archive), "--dest", str(dest), "--tar", tar, "--data-root", str(tmp_path / "data")])
    except SystemExit as exc:
        code = exc.code
    assert code == cli.EXIT_PROTOCOL_NOT_FROZEN and not dest.exists()


def test_extraction_writes_only_the_external_files_and_rows(tmp_path, monkeypatch, capsys):
    tar = f2.find_tar()
    if tar is None:
        pytest.skip("no tar executable")
    archive = f2.make_mms2_archive(tmp_path / "archive")
    dest = tmp_path / "raw" / "mms2"
    monkeypatch.setattr(stages, "protocol_frozen", lambda tag="", **k: tag == "protocol-v6.0.1")   # mocked tag check
    try:
        cli.main(["extract-mms2", "--archive", str(archive), "--dest", str(dest), "--tar", tar, "--data-root", str(tmp_path / "data")])
    except SystemExit as exc:
        pytest.fail(f"extract-mms2 exited {exc.code}: {capsys.readouterr().out}")
    receipt = json.loads((dest / mms2.RECEIPT_JSON).read_text(encoding="utf-8"))
    assert receipt["lock"] == "protocol_frozen" and receipt["list_file"]["entries"] == 720 and receipt["byte_check"]["matches_listing"]
    files = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
    data_files = {f for f in files if f.endswith(".nii.gz")}
    assert len(data_files) == 720 and {Path(f).name.split("_", 1)[0] for f in data_files} == set(EXTERNAL)
    assert not any("LA_" in f or "CINE" in f for f in data_files)
    assert files - data_files == {mms2.FILE_MANIFEST, mms2.RECEIPT_JSON, mms2.INFO_EXTERNAL, fx.MARKER}
    listing = {e.path: e.size for e in mms2.list_archive(archive, tar)[0]}
    assert all((dest / f).stat().st_size == listing[f] for f in data_files)
    manifest = mms2.load_file_manifest(dest)                     # refuses any sealed subject
    assert len(manifest) == 721 and set(manifest.loc[manifest["role"] != "info_csv", "case_id"]) == {mms2.case_key(s) for s in EXTERNAL}
    layout = receipt["layout"]
    assert layout["listing_entries_match"] and layout["info_csv_matches"] and layout["members_matching_template"] == 720
    info = pd.read_csv(dest / mms2.INFO_EXTERNAL, dtype=str)
    assert sorted(info["SUBJECT_CODE"].map(mms2.normalize_subject_code)) == EXTERNAL
    assert receipt["release_csv"]["rows"] == 360 and receipt["release_csv"]["rows_external"] == 180
    for path in dest.rglob("*"):                                  # no sealed subject name or CSV row anywhere in the area
        if path.is_file():
            assert not any(f"/{s}/" in f"/{path.relative_to(dest).as_posix()}" for s in SEALED)
    record = json.loads((tmp_path / "data" / "manifests" / "extraction_MMS2.json").read_text(encoding="utf-8"))
    assert record["action"] == "RAN"                              # the dated extraction record of SPECIFICATION 14.6
    started = stages.parse_utc(receipt["extraction_started_utc"])
    assert started is not None and started <= stages.parse_utc(receipt["extraction_completed_utc"])
    assert stages.parse_utc(record["first_extraction_started_utc"]) == started
    again = mms2.extract_external(archive, dest, cohort_cfg=_cfg(), tar=tar, require_unlocked=lambda: "protocol_frozen")
    assert again["action"] == "SKIPPED"


def test_extraction_fails_closed_on_an_unknown_csv_column(tmp_path):
    tar = f2.find_tar()
    if tar is None:
        pytest.skip("no tar executable")
    rows = [dict(f2.info_row(sid, i), CENTRE="1") for i, sid in enumerate(mms2.SUBJECT_IDS)]
    archive = f2.make_mms2_archive(tmp_path / "archive", info_rows=rows, info_columns=f2.INFO_COLUMNS + ["CENTRE"])
    dest = tmp_path / "raw" / "mms2"
    with pytest.raises(mms2.HumanGateError, match="CENTRE"):
        mms2.extract_external(archive, dest, cohort_cfg=_cfg(), tar=tar, require_unlocked=lambda: "protocol_frozen")
    assert not dest.exists() or not any(dest.rglob("*.nii.gz"))   # nothing extracted before the CSV contract passes


@pytest.mark.parametrize("argv", [["stage-sources", "--dataset", "mms2"], ["validate-cohort", "--dataset", "mms2"],
                                  ["build-semantics", "--dataset", "mms2"], ["extract-formal", "--dataset", "mms2"],
                                  ["evaluate", "--split", "mms2_external", "--no-fit"], ["audit-shift", "--dataset", "mms2"],
                                  ["summarize-predictor", "--split", "mms2_external"]])
def test_mms2_compute_stages_exit_3_before_the_tag(argv, tmp_path, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(stages, "protocol_frozen", lambda tag="protocol-v6.0.0", **k: seen.append(tag) or False)
    with pytest.raises(SystemExit) as info:
        cli.main(argv + ["--run-id", "v6-guard-20261003-0900", "--artifact-root", str(tmp_path / "a"), "--data-root", str(tmp_path / "d")])
    assert info.value.code == cli.EXIT_PROTOCOL_NOT_FROZEN
    assert "protocol-v6.0.1" in seen and "REFUSED" in capsys.readouterr().out


def test_mms2_lock_override_only_for_synthetic_fixtures(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "protocol_frozen", lambda *a, **k: False)
    root = f2.make_mms2_root(tmp_path / "mms2", [EXTERNAL[0]])
    ctx = _context(tmp_path, allow=True, sources={"mms2": root})
    stages.stage_sources(ctx, "mms2")
    assert stages.require_mms2_unlocked(ctx, "test") == "synthetic_override"
    (root / fx.MARKER).unlink()
    with pytest.raises(ProtocolFreezeError, match="synthetic"):
        stages.require_mms2_unlocked(ctx, "test")
    monkeypatch.setattr(stages, "protocol_frozen", lambda tag="", **k: tag == "protocol-v6.0.1")
    assert stages.require_mms2_unlocked(ctx, "test") == "protocol_frozen"


def test_mms2_lock_requires_a_signed_annotated_tag(tmp_path, monkeypatch):
    """An unsigned ``git tag -a protocol-v6.0.1`` never unlocks M&Ms-2 (git answers are simulated; no repository is touched)."""
    import subprocess

    unsigned = "object 0123\ntype commit\ntag protocol-v6.0.1\ntagger Owner <o@example.org> 1 +0000\n\ntest\n"
    signed = unsigned + "-----BEGIN SSH SIGNATURE-----\nU1NIU0lH\n-----END SSH SIGNATURE-----\n"

    def fake_git(body):
        def run(args, **kwargs):
            out = {"-t": "tag\n", "tag": body}.get(args[3], "") if args[2] == "cat-file" else ""
            return subprocess.CompletedProcess(args, 0, out if kwargs.get("text") else out.encode(), "" if kwargs.get("text") else b"")
        return run

    ctx = _context(tmp_path)
    monkeypatch.setattr(stages.subprocess, "run", fake_git(unsigned))
    assert stages.protocol_frozen("protocol-v6.0.1") and not stages.protocol_frozen("protocol-v6.0.1", signed=True)
    with pytest.raises(ProtocolFreezeError, match="unsigned"):
        stages.require_mms2_unlocked(ctx, "test")
    monkeypatch.setattr(stages.subprocess, "run", fake_git(signed))
    assert stages.protocol_frozen("protocol-v6.0.1", signed=True) and stages.require_mms2_unlocked(ctx, "test") == "protocol_frozen"
    assert stages.tag_has_signature(signed.replace("SSH", "PGP")) and not stages.tag_has_signature(unsigned + "-----BEGIN SSH SIGNATURE-----\n")


# ----------------------------------------------------------------------------- duplicate rule


def test_duplicate_rule_excludes_matches_against_acdc_mms1_and_within_mms2(tmp_path):
    acdc = fx.make_acdc_source(tmp_path / "src" / "acdc", fx.acdc_patients(1))
    mms = fx.make_mms_source(tmp_path / "src" / "mms")
    study = load_study()
    run = RunRoot("v6-mms2dup-20261003-0900", tmp_path / "artifacts", subfolders=study.study["artifact_subfolders"])
    ctx = StageContext(study=study, run=run, data_root=tmp_path / "data", n_jobs=1, allow_unfrozen_synthetic=True,
                       sources={"acdc": acdc, "mms": mms}, source_manifests={"ACDC": fx.acdc_manifest(acdc), "MMS": fx.mms_manifest(mms)},
                       mms_metadata=fx.mms_metadata(mms))
    assert stages.stage_sources(ctx, "acdc").ok and stages.stage_sources(ctx, "mms").ok
    a_dup, m_dup, w1, w2, clean = EXTERNAL[:5]
    acdc_case = load_staged_case("ACDC", "patient001", ctx.staged_root)       # unit-scaled ACDC frame files
    mms_case = load_staged_case("MMS", "B2B2B2", ctx.staged_root)             # voxel-scaled, same affine as f2.VOXEL_AFFINE
    assert np.allclose(mms_case.affine, f2.VOXEL_AFFINE)
    shared = f2.case_arrays(w1)
    arrays = {a_dup: {"ed_image": acdc_case.ed_image, "ed_mask": np.ones(acdc_case.ed_image.shape, np.uint8),
                      "es_image": acdc_case.es_image, "es_mask": np.ones(acdc_case.es_image.shape, np.uint8)},
              m_dup: {"ed_image": mms_case.ed_image, "ed_mask": np.ones(mms_case.ed_image.shape, np.uint8),
                      "es_image": mms_case.es_image + 1.0, "es_mask": np.ones(mms_case.es_image.shape, np.uint8)},
              w1: shared, w2: {**f2.case_arrays(w2), "ed_image": shared["ed_image"]}, clean: f2.case_arrays(clean)}
    root = tmp_path / "src" / "mms2"
    cases = {sid: f2.write_case(root, sid, arrays[sid], geometry="unit" if sid == a_dup else "voxel") for sid in (a_dup, m_dup, w1, w2, clean)}
    f2.finish_root(root, cases, [f2.info_row(s, i) for i, s in enumerate(cases)])
    ctx.sources["mms2"] = root
    staged = stages.stage_sources(ctx, "mms2")
    assert staged.ok, staged.failures
    report = validate_cohort_mms2(ctx)
    exclusions = {d["sample_id"]: d["reason_code"] for d in report["duplicate_exclusions"]}
    assert exclusions == {f"MMS2:MMS2_{a_dup}": "DUPLICATE_IMAGE_OF_ACDC_CASE", f"MMS2:MMS2_{m_dup}": "DUPLICATE_IMAGE_OF_MMS1_CASE",
                          f"MMS2:MMS2_{w1}": "DUPLICATE_IMAGE_WITHIN_MMS2", f"MMS2:MMS2_{w2}": "DUPLICATE_IMAGE_WITHIN_MMS2"}
    checks = {c["check"]: c for c in report["checks"]}
    assert checks["mms2_duplicates_excluded"]["passed"] and checks["mms2_no_sealed_subject"]["passed"]
    assert checks["mms2_split_manifest_reproduces_rule"]["passed"] and checks["mms2_split_manifest_sha256_matches_receipt"]["passed"]
    assert not checks["mms2_external_size"]["passed"] and checks["mms2_external_size"]["waived"]
    assert report["passed"], [c for c in report["checks"] if not c["passed"]]
    cohort = pd.read_parquet(run.path / "contracts" / "cohort_MMS2.parquet").set_index("case_id")
    assert list(cohort.index[cohort["mms2_external"]]) == [mms2.case_key(clean)]
    dup = mms2.case_key(a_dup)
    assert cohort.at[dup, "duplicate_of"] == "ACDC:patient001" and json.loads(cohort.at[dup, "sealed_link"])["partition"] == "mms2_sealed"
    split = pd.read_parquet(run.path / "contracts" / "split_manifest_MMS2.parquet")
    assert dict(zip(split["case_id"], split["partition"]))[mms2.case_key(clean)] == "mms2_external" and (split["partition"] == "excluded").sum() == 4
    links = json.loads((run.path / "contracts" / "mms2_duplicates.json").read_text(encoding="utf-8"))["sealed_links"]
    assert len(links) == 4 and all(link["status"] == "NOT_CHECKABLE_UNDER_V6" for link in links)
    assert not (run.path / "contracts" / "split_manifest.parquet").exists()      # the v6.0.0 contracts are never written here


def test_failed_mms2_validation_writes_no_contract_or_record_and_blocks_later_stages(tmp_path):
    """The duplicate exclusion is a leakage control: with ACDC not staged the validation fails (exit 6 path) even without
    --strict, writes neither contract nor record, the later M&Ms-2 stages refuse, and staging ACDC then repairs it."""
    mms = fx.make_mms_source(tmp_path / "src" / "mms")
    study = load_study()
    run = RunRoot("v6-mms2fail-20261003-0900", tmp_path / "artifacts", subfolders=study.study["artifact_subfolders"])
    ctx = StageContext(study=study, run=run, data_root=tmp_path / "data", n_jobs=1, allow_unfrozen_synthetic=True,
                       sources={"mms": mms}, source_manifests={"MMS": fx.mms_manifest(mms)}, mms_metadata=fx.mms_metadata(mms))
    assert stages.stage_sources(ctx, "mms").ok
    ctx.sources["mms2"] = f2.make_mms2_root(tmp_path / "src" / "mms2", EXTERNAL[:3])
    assert stages.stage_sources(ctx, "mms2").ok
    contracts = run.path / "contracts"
    with pytest.raises(stages.CohortValidationError) as info:
        validate_cohort_mms2(ctx, strict=False)                                # always strict with a run id
    failed = {c["check"] for c in info.value.report["checks"] if not c["passed"]}
    assert "mms2_duplicate_reference_cohorts_staged" in failed
    assert not any((contracts / n).exists() for n in ("cohort_MMS2.parquet", "split_manifest_MMS2.parquet", "mms2_duplicates.json"))
    report = json.loads((contracts / "cohort_validation_MMS2.json").read_text(encoding="utf-8"))
    assert report["passed"] is False and report["contracts_written"] is False
    assert run.read_record("validate-cohort:MMS2") is None
    for call in (lambda: stages.build_semantics(ctx, "mms2"), lambda: stages.extract_formal(ctx, "mms2"),
                 lambda: stages.require_mms2_validated(ctx, "evaluate --split mms2_external")):
        with pytest.raises(stages.BundleError, match="validate-cohort --dataset mms2"):
            call()
    acdc = fx.make_acdc_source(tmp_path / "src" / "acdc", fx.acdc_patients(1))
    ctx.sources["acdc"] = acdc
    ctx.source_manifests["ACDC"] = fx.acdc_manifest(acdc)
    assert stages.stage_sources(ctx, "acdc").ok
    assert validate_cohort_mms2(ctx)["passed"] and run.read_record("validate-cohort:MMS2") is not None
    assert stages.require_mms2_validated(ctx, "test") == run.record_path("validate-cohort:MMS2")
    pd.read_parquet(contracts / "split_manifest_MMS2.parquet").assign(partition="mms2_external").to_parquet(contracts / "split_manifest_MMS2.parquet")
    with pytest.raises(stages.BundleError, match="changed after"):            # a rewritten contract is never used
        stages.require_mms2_validated(ctx, "test")
