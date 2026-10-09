"""Backend qualification on synthetic tensors only (protocol 2.1 Section H item 3; Modal assessment Section 4 C2-C3).

:func:`run_qualification` writes a synthetic C3 cache and a seeded random ResNet-18 (no cohort data, no downloaded
weights) and reports:

* throughput: training (forward + backward + AdamW step, batches of four patients, with the D4 augmentation) and
  inference slices per second (the D4 test-time symmetrization ``d4_mean_8`` of every reported output: eight backbone
  passes per slice, all counted) for both unfrozen-block settings;
* determinism: two identical fits must have identical deterministic content (epochs, selected epoch, validation
  outputs, final state hash);
* resume equivalence: a fit interrupted after epoch 1 and resumed must equal the uninterrupted fit;
* IEEE FP32: the C9 state is asserted, and a float32 matmul and convolution on the device are compared with float64
  references (TF32 would show relative errors near 1e-3).

``scripts/cloud/modal_train.py --qualify`` runs it on the pinned Modal GPU; the tests run it on CPU with tiny shapes.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

QUALIFY_IEEE_REL_TOL = 1e-4


def _qualification_config(cfg: Mapping[str, Any], epochs: int) -> dict[str, Any]:
    out = copy.deepcopy(dict(cfg))
    out["epochs"] = {"max": int(epochs)}
    out["early_stopping"] = {**dict(out["early_stopping"]), "min_epoch": int(epochs) + 1, "patience": int(epochs) + 1}
    return out


def _ieee_check(device: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    gen = torch.Generator().manual_seed(20261011)
    a, b = torch.randn(512, 512, generator=gen), torch.randn(512, 512, generator=gen)
    ref = (a.double() @ b.double())
    got = (a.to(device) @ b.to(device)).to("cpu").double()
    mm_err = float((got - ref).abs().max() / ref.abs().max())
    x, w = torch.randn(4, 64, 28, 28, generator=gen), torch.randn(64, 64, 3, 3, generator=gen)
    cref = F.conv2d(x.double(), w.double(), padding=1)
    cgot = F.conv2d(x.to(device), w.to(device), padding=1).to("cpu").double()
    conv_err = float((cgot - cref).abs().max() / cref.abs().max())
    return {"matmul_max_rel_error": mm_err, "conv_max_rel_error": conv_err,
            "passed": mm_err < QUALIFY_IEEE_REL_TOL and conv_err < QUALIFY_IEEE_REL_TOL}


def _benchmark(items: list[dict[str, Any]], weights: Path, device: Any, cfg: Mapping[str, Any], steps: int) -> dict[str, Any]:
    import torch

    from .data import augment_batch, batch_to, collate_patients
    from .dihedral import D4_TTA, f9_forward_tta, tta_elements
    from .fit_spec import Candidate
    from .model import build_model

    out: dict[str, Any] = {}
    passes = len(tta_elements(D4_TTA))
    batch_cpu = collate_patients(items[:4])
    n_slices = int(batch_cpu["ed_mask"].sum() + batch_cpu["es_mask"].sum())
    sync = torch.cuda.synchronize if torch.device(device).type == "cuda" else (lambda: None)
    for unfreeze in ("layer4", "layer3+layer4"):
        model = build_model(Candidate(unfreeze, 0.0001), weights, seed=20261011).to(device)
        optimizer = torch.optim.AdamW(model.parameter_groups(0.0001, 10.0), lr=0.0001, weight_decay=0.0001)
        rng = np.random.default_rng(0)

        def step(model=model, optimizer=optimizer, rng=rng) -> None:
            batch = augment_batch(batch_to(batch_cpu, device), rng, cfg["augmentation"], spacing_mm=float(cfg["input"]["inplane_spacing_mm"]))
            model.train()
            logits, _ = model.forward_batch(batch)
            loss = -torch.log_softmax(logits, dim=1)[:, 0].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        for _ in range(2):
            step()
        sync()
        t0 = time.perf_counter()
        for _ in range(int(steps)):
            step()
        sync()
        train_s = time.perf_counter() - t0
        model.eval()
        batch = batch_to(batch_cpu, device)
        with torch.no_grad():
            f9_forward_tta(model, batch, tta=D4_TTA)
            sync()
            t0 = time.perf_counter()
            for _ in range(int(steps)):
                f9_forward_tta(model, batch, tta=D4_TTA)
            sync()
        infer_s = time.perf_counter() - t0
        out[unfreeze] = {"train_slices_per_s": n_slices * int(steps) / train_s,
                         "inference_slices_per_s": n_slices * passes * int(steps) / infer_s, "inference_tta": D4_TTA,
                         "inference_passes_per_slice": passes, "slices_per_batch": n_slices, "steps": int(steps)}
    return out


def run_qualification(workdir: str | Path, device: Any = "auto", *, config: Mapping[str, Any] | None = None, size: int = 224,
                      n_per_class: int = 3, slices: tuple[int, int] = (8, 10), epochs: int = 3, bench_steps: int = 10) -> dict[str, Any]:
    """Synthetic-only qualification report (see the module docstring); ``passed`` is the conjunction of all checks."""

    from .f9_config import load_f9_config, training_config
    from .fit_spec import PRIMARY_SEED, Candidate, FitSpec
    from .numerics import configure_numerics, environment_record
    from .synthetic_cache import write_random_resnet18, write_synthetic_cache
    from .train import WeightsRef, resolve_device, train_fit

    numerics = configure_numerics()
    dev = resolve_device(device)
    work = Path(workdir)
    cfg = _qualification_config(config if config is not None else training_config(load_f9_config()), epochs)
    manifest = write_synthetic_cache(work / "cache", n_per_class=n_per_class, slices=slices, size=size)
    weights_path, weights_sha = write_random_resnet18(work / "weights" / "resnet18_random.pt")
    weights = WeightsRef(path=str(weights_path), sha256=weights_sha, name="synthetic_random_init")
    from .data import load_npz, npz_path, read_manifest

    ids = sorted(read_manifest(manifest)["sample_id"])
    items = [load_npz(npz_path(manifest.parent, sid)) for sid in ids]
    report: dict[str, Any] = {"synthetic_only": True, "device": str(dev), "numerics": numerics, "settings":
                              {"size": size, "n_per_class": n_per_class, "slices": list(slices), "epochs": epochs, "bench_steps": bench_steps}}
    report["ieee_fp32"] = _ieee_check(dev)
    report["benchmark"] = _benchmark(items, weights_path, dev, cfg, bench_steps)
    spec = FitSpec(purpose="f9", seed=PRIMARY_SEED, outer_fold="final", fitting_patients=tuple(ids), selection_fold=1,
                   candidate=Candidate("layer3+layer4", 0.0001), refit_epochs=None)
    kwargs = {"config": cfg, "weights": weights, "backend": {"name": "qualification"}}
    first = train_fit(spec, work / "cache", work / "fits" / "a", dev, **kwargs)
    second = train_fit(spec, work / "cache", work / "fits" / "b", dev, **kwargs)
    interrupted = train_fit(spec, work / "cache", work / "fits" / "c", dev, stop_after_epoch=1, **kwargs)
    resumed = train_fit(spec, work / "cache", work / "fits" / "c", dev, **kwargs)
    report["determinism"] = {"passed": first.deterministic_view() == second.deterministic_view(),
                             "state_sha256": [first.state_sha256, second.state_sha256]}
    report["resume"] = {"passed": interrupted.status == "INTERRUPTED" and resumed.resume.get("resumed_from_epoch") == 2
                        and resumed.deterministic_view() == first.deterministic_view(),
                        "state_sha256": [first.state_sha256, resumed.state_sha256], "resumed_from_epoch": resumed.resume.get("resumed_from_epoch")}
    report["tf32_disabled"] = not numerics["tf32_enabled"]
    report["fit_seconds"] = first.timings.get("seconds_total")
    report["environment"] = environment_record(dev)
    report["passed"] = bool(report["ieee_fp32"]["passed"] and report["determinism"]["passed"] and report["resume"]["passed"]
                            and report["tf32_disabled"])
    return report
