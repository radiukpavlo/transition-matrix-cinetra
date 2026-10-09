from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .io import PROJECT_ROOT, read_yaml, resolve_path
from .paths import resolve_recorded_path


def load_config(path: str | Path = "configs/legacy/pipeline/default.yaml") -> dict[str, Any]:
    # Recorded commands still name configs/pipeline/...; resolve_recorded_path
    # maps those to configs/legacy/... when the old path is missing (ADR-0004).
    resolved = resolve_recorded_path(path)
    cfg = read_yaml(resolved)
    cfg["_config_path"] = str(resolve_path(resolved))
    return cfg


def load_clinical_conventions(path: str | Path = "configs/clinical/clinical_conventions.yaml") -> dict[str, Any]:
    return read_yaml(path)


# ----------------------------------------------------------------------------- protocol 6.0.0 study config

# Top-level keys added to a protocol-6.0.0 configuration file by a later amendment (protocol-v6.0.1: the M&Ms-2 cohort;
# the D20 test-time symmetrization ``deep_encoder.tta``, owner issue O1).  They are excluded from the digest of the
# whole-file section, so the stage records and the frozen bundle of a v6.0.0 run stay valid; the stages that read them
# name the sub-section explicitly (for example ``cohorts.mms2`` or ``deep_encoder.tta``).
AMENDMENT_SCOPED_KEYS: dict[str, tuple[str, ...]] = {"cohorts": ("mms2",), "splits": ("mms2",), "deep_encoder": ("tta",)}


def _normalized_sha256(path: Path) -> str:
    """SHA-256 of a text file with CRLF normalized to LF (host checkouts and the container agree)."""

    import hashlib

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _read_any(path: Path) -> Any:
    import csv
    import json

    if path.suffix in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".env":
        return _read_env_file(path)
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"unsupported config file type: {path}")


@dataclass(frozen=True)
class StudyConfig:
    """All protocol-6.0.0 configs and registry files, with per-file SHA-256 and a combined config hash.

    ``configs`` is keyed by the path below ``configs/`` without extension (``study_v6``, ``cohorts``, ``splits``,
    ``formal``, ``transition``, ``wedd``, ``rule_policy``, ``equivariance``, ``deep_encoder``, ``bootstrap``,
    ``runtime``, ``source_bindings/acdc``, ``source_bindings/mms``).  ``file_hashes`` maps repository-relative
    POSIX paths to LF-normalized SHA-256; ``config_hash`` is the SHA-256 of the sorted ``path<TAB>sha256`` lines.
    """

    root: Path
    path: Path
    study: dict[str, Any]
    configs: dict[str, Any]
    registry: dict[str, Any]
    reason_codes: dict[str, Any]
    pattern_templates: dict[str, Any]
    estimator_partitions: list[dict[str, str]]
    file_hashes: dict[str, str]
    config_hash: str

    @property
    def cohorts(self) -> dict[str, Any]:
        return self.configs["cohorts"]

    @property
    def splits(self) -> dict[str, Any]:
        return self.configs["splits"]

    @property
    def formal(self) -> dict[str, Any]:
        return self.configs["formal"]

    @property
    def transition(self) -> dict[str, Any]:
        return self.configs["transition"]

    @property
    def wedd(self) -> dict[str, Any]:
        return self.configs["wedd"]

    @property
    def rule_policy(self) -> dict[str, Any]:
        return self.configs["rule_policy"]

    @property
    def bootstrap(self) -> dict[str, Any]:
        return self.configs["bootstrap"]

    @property
    def seeds(self) -> dict[str, Any]:
        return dict(self.study.get("seeds", {}))

    @property
    def freeze_tag(self) -> str:
        return str(self.study["freeze"]["tag"])

    # --- protocol-v6.0.1 (owner decision F): the protocol tag a run is executed under
    @property
    def protocol_tags(self) -> list[str]:
        """The freeze tag followed by the amendment tags (``freeze.amendment_tags``), oldest first."""

        return [self.freeze_tag, *[str(t) for t in (self.study["freeze"].get("amendment_tags") or [])]]

    @property
    def current_protocol_tag(self) -> str:
        """The newest protocol tag this configuration implements (recorded in FROZEN.json by new runs)."""

        return self.protocol_tags[-1]
    # --- end protocol-v6.0.1

    def source_binding(self, cohort: str) -> dict[str, Any]:
        return self.configs[f"source_bindings/{cohort}"]

    def label_map(self, cohort: str) -> dict[int, int]:
        return {int(k): int(v) for k, v in self.cohorts[cohort]["label_map"].items()}

    def features(self) -> dict[str, dict[str, Any]]:
        return {str(f["name"]): dict(f) for f in self.registry.get("features", [])}

    def transition_targets(self) -> list[str]:
        """Registry features with ``transition_role`` candidate or candidate_secondary (registry order)."""

        return [n for n, f in self.features().items() if f.get("transition_role") in {"candidate", "candidate_secondary"}]

    def primary_family(self) -> list[str]:
        family = self.transition["targets"]["primary_family"]
        return list(self.registry["primary_families"][family])

    def rule_candidates(self) -> list[str]:
        return list(self.registry["primary_rulebook_antecedent_candidates"])

    def get(self, dotted: str, default: Any = None) -> Any:
        """``get("splits.acdc.acdc_dev.outer_folds.n_folds")``; the first component names a config or ``registry``."""

        head, _, rest = dotted.partition(".")
        value: Any = {"registry": self.registry, "reason_codes": self.reason_codes, "study": self.study}.get(head, self.configs.get(head))
        for part in [p for p in rest.split(".") if p]:
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return default if value is None else value

    def hash_manifest(self) -> dict[str, Any]:
        return {"config_hash": self.config_hash, "files": dict(self.file_hashes)}

    def section(self, dotted: str) -> Any:
        """The parsed value of ``dotted`` (``get`` syntax); a missing section raises :class:`KeyError`."""

        head, _, rest = dotted.partition(".")
        value: Any = {"registry": self.registry, "reason_codes": self.reason_codes, "study": self.study}.get(head, self.configs.get(head))
        if value is None and head not in self.configs:
            raise KeyError(f"unknown configuration section {dotted!r}")
        for part in [p for p in rest.split(".") if p]:
            if not isinstance(value, dict) or part not in value:
                raise KeyError(f"unknown configuration section {dotted!r}")
            value = value[part]
        return value

    def sections_digest(self, sections: list[str] | tuple[str, ...]) -> dict[str, str]:
        """SHA-256 of the canonical JSON of each parsed section (comment-only edits do not change it).

        A whole-file section (``cohorts``, ``splits``, ``deep_encoder``) is hashed without its
        :data:`AMENDMENT_SCOPED_KEYS`: those keys were added by a later amendment, are read only by the stages that name
        them explicitly (``cohorts.mms2``, ``deep_encoder.tta``), and
        so never change the digests recorded by protocol-6.0.0 stages or by a frozen bundle.
        """

        import hashlib
        import json

        def value(dotted: str) -> Any:
            section = self.section(dotted)
            scoped = AMENDMENT_SCOPED_KEYS.get(dotted)
            if scoped and isinstance(section, dict):
                section = {k: v for k, v in section.items() if k not in scoped}
            return section

        return {s: hashlib.sha256(json.dumps(value(s), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
                for s in sorted(set(sections))}

    def sections_hash(self, sections: list[str] | tuple[str, ...]) -> str:
        """Configuration hash of a stage: the hash of :meth:`sections_digest` (Decision D23, per-stage skips)."""

        import hashlib

        digest = self.sections_digest(sections)
        return hashlib.sha256("\n".join(f"{k}\t{v}" for k, v in digest.items()).encode("utf-8")).hexdigest()


def load_study(path: str | Path = "configs/study_v6.yaml", root: str | Path | None = None) -> StudyConfig:
    """Load ``configs/study_v6.yaml`` and every file it lists (``config_files`` and ``registry_files``).

    Listed files resolve against ``root`` (default the repository).  An absolute study path outside the repository
    (a copied configuration tree, as in the synthetic end-to-end test) resolves against the folder that contains its
    ``configs/`` directory, so the copy's configs and registry are read.
    """

    import hashlib

    base = Path(root) if root is not None else PROJECT_ROOT
    if root is None and Path(path).is_absolute():
        try:
            Path(path).resolve().relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            base = Path(path).resolve().parents[1]
    study_path = Path(path) if Path(path).is_absolute() else base / path
    study = _read_any(study_path)
    config_files = list(dict.fromkeys([str(Path(path).as_posix()) if not Path(path).is_absolute() else study_path.relative_to(base).as_posix(),
                                       *study.get("config_files", [])]))
    registry_files = list(study.get("registry_files", []))
    hashes: dict[str, str] = {}
    configs: dict[str, Any] = {}
    for rel in config_files:
        file = base / rel
        key = Path(rel).relative_to("configs").with_suffix("").as_posix() if rel.startswith("configs/") else Path(rel).stem
        configs[key] = study if file.resolve() == study_path.resolve() else _read_any(file)
        hashes[Path(rel).as_posix()] = _normalized_sha256(file)
    loaded: dict[str, Any] = {}
    for rel in registry_files:
        file = base / rel
        loaded[Path(rel).stem] = _read_any(file)
        hashes[Path(rel).as_posix()] = _normalized_sha256(file)
    combined = hashlib.sha256("\n".join(f"{k}\t{v}" for k, v in sorted(hashes.items())).encode("utf-8")).hexdigest()
    return StudyConfig(
        root=base, path=study_path, study=study, configs=configs,
        registry=loaded.get("semantic_registry", {}), reason_codes=loaded.get("reason_codes", {}),
        pattern_templates=loaded.get("pattern_templates", {}),
        estimator_partitions=loaded.get("estimator_partition_registry", []),
        file_hashes=dict(sorted(hashes.items())), config_hash=combined,
    )


def cfg_path(cfg: dict[str, Any], dotted: str, default: str | Path | None = None) -> Path:
    value: Any = cfg
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            if default is None:
                raise KeyError(dotted)
            return resolve_path(default, PROJECT_ROOT)
        value = value[part]
    return resolve_path(value, PROJECT_ROOT)
