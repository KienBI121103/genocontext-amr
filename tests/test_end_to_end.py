"""Tiny synthetic cohort through the CLI: run, e0, stack, explain, candidates and compare (seconds, CPU)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from genocontext.cli import main
from genocontext.features.profile import Node, Profile, save_profile
from tests.conftest import ROOT

DRUGS = ("imipenem", "ciprofloxacin")
TARGETS = ("GyrA", "ParC", "OmpK35", "OmpK36", "OmpK37", "PhoE", "RamR", "AcrR", "MarR", "SoxR", "NfsA", "NfsB",
           "RibE")


def _profile(isolate: str, kpc: bool, qrdr: bool, rng: np.random.Generator) -> Profile:
    nodes = [Node(f"target:{t}", "target", "intact", ["other"], [0, 0, 0], [[0, 100 + i]], [f"{isolate}_{t}"])
             for i, t in enumerate(TARGETS)]
    nodes += [Node(f"fam:F{k}", "fam", "present", ["other"], [0, 0, 0], [[0, k]], [f"{isolate}_F{k}"])
              for k in range(12) if rng.random() < 0.5]
    if kpc:
        nodes.append(Node("amr:blaKPC-3", "amr", "present", ["carbapenemase"], [0, 0, 1], [[0, 3]],
                          [f"{isolate}_K"], "amrfam:blaKPC"))
    if qrdr:
        nodes.append(Node("res:GyrA_S83I", "res", "present", ["fq_target"], [0, 0, 0], [[0, 100]],
                          [f"{isolate}_GyrA"]))
    return Profile(isolate, {"n_contigs": 10.0}, nodes)


def _cohort(root: Path, n: int = 60) -> Path:
    rng = np.random.default_rng(0)
    ids = [f"iso{i}" for i in range(n)]
    kpc, qrdr = rng.random(n) < 0.5, rng.random(n) < 0.5
    (root / "profiles").mkdir()
    for i, isolate in enumerate(ids):
        save_profile(_profile(isolate, kpc[i], qrdr[i], rng), root / "profiles" / f"{isolate}.json.gz")
    labels = pd.DataFrame({"id": ids, "Biosample": [f"B{i}" for i in range(n)],
                           "imipenem": kpc.astype(float), "ciprofloxacin": qrdr.astype(float)})
    labels.loc[::7, "ciprofloxacin"] = np.nan
    labels.to_csv(root / "labels.csv", index=False)
    pd.DataFrame({"id": ids, "assembly_path": [f"/none/{i}.fna" for i in ids]}).to_csv(root / "manifest.csv",
                                                                                        index=False)
    split = root / "splits" / "cg_blocked" / "seed0"
    split.mkdir(parents=True)
    groups = [f"CG{i % 15}" for i in range(n)]
    pd.DataFrame({"isolate_id": ids, "group": groups, "outer": ["test" if i >= 45 else "train" for i in range(n)],
                  "inner_fold": [i % 5 if i < 45 else -1 for i in range(n)]}).to_csv(split / "partition.tsv",
                                                                                     sep="\t", index=False)
    (root / "splits" / "split_protocol.json").write_text("{}")
    pd.DataFrame({"isolate_id": ids, "group": groups}).to_csv(root / "splits" / "clonal_groups.tsv", sep="\t",
                                                              index=False)
    config = {"paths": {"labels": str(root / "labels.csv"), "manifest": str(root / "manifest.csv"),
                        "splits": str(root / "splits"), "profiles": str(root / "profiles"),
                        "artifacts": str(root / "artifacts"),
                        "knowledge": str(ROOT / "knowledge/mechanisms.yaml"),
                        "references": str(ROOT / "knowledge/references/kp_targets.faa")},
              "features": {"min_prevalence": 2, "max_families": 20},
              "model": {"hidden": 8, "query": 4},
              "training": {"epochs": 3, "patience": 2, "batch_size": 16, "ensemble_seeds": 1, "threads": 1,
                           "workers": 1}}
    path = root / "config.yaml"
    path.write_text(json.dumps(config))  # JSON is valid YAML
    return path


def test_cli_end_to_end(tmp_path: Path) -> None:
    config = str(_cohort(tmp_path))
    common = ["--config", config, "--scheme", "cg_blocked"]
    main(["run", *common, "--model", "kg", "lgbm_genestate"])
    main(["run", *common, "--model", "kg", "--variant", "shuffled_labels", "zeroshot=imipenem"])
    main(["e0", *common, "--model", "lgbm_genestate"])
    main(["stack", *common, "--model", "kg", "lgbm_genestate"])
    main(["explain", *common, "--model", "kg"])
    main(["candidates", *common])
    main(["compare", *common, "--reference", str(tmp_path / "missing.json")])
    runs = tmp_path / "artifacts" / "runs" / "cg_blocked" / "seed0" / "all"
    assert {p.name for p in runs.iterdir()} >= {"kg", "lgbm_genestate", "kg+lgbm_genestate",
                                                "kg@shuffled_labels", "kg@zeroshot=imipenem"}
    explain = runs / "kg" / "B" / "explain"
    nodes = pd.read_csv(explain / "node_attributions.tsv.gz", sep="\t")
    assert nodes["loci"].notna().any()
    edges = pd.read_csv(explain / "edge_weights.tsv", sep="\t")
    assert set(edges.columns) == {"mechanism", "drug", "weight", "prior"}
    results = tmp_path / "artifacts" / "results"
    metrics = pd.read_csv(results / "metrics.tsv", sep="\t")
    lgbm = metrics[(metrics["model"] == "lgbm_genestate") & (metrics["drug"] == "imipenem")]
    assert lgbm["auroc"].iloc[0] > 0.9  # blaKPC fully determines the synthetic imipenem phenotype
    assert (results / "seed_summary.tsv").exists() and (results / "low_data.tsv").exists()
    assert json.loads((tmp_path / "artifacts" / "e0" / "e0_decision.json").read_text())["per_model"]
