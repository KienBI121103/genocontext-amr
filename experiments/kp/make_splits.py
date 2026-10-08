"""Write the GenoContext v2 K. pneumoniae split tiers (run once from the terminal; never imported).

Tiers (each with 10 seeds and isolate-level ``partition.tsv`` files):
  amrgnn_fixed   AMR-GNN per-drug splits; train = train ∪ val, test kept verbatim; inner fold 0 = original val.
  random_grouped BioSample-deduplicated isolates, balanced 5-fold × 2 repeats (seed k = repeat k//5, fold k%5).
  cg_blocked     same builder over clonal groups (one-hop single-locus-variant stars around large STs).

Usage:
  python experiments/kp/make_splits.py --out /mnt/nas/.../splits_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

KP = Path("/mnt/nas/earth/KienNM/amrgnn_multi/kpneumoniae")
FOLDS, REPEATS, SEEDS = 5, 2, 10
MIN_MINORITY_TEST, MIN_MINORITY_VAL = 10, 5
BIG_GROUP_FRACTION = 0.05  # clonal groups above this share are never the protocol-A validation fold


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_inputs(labels_path: Path, mlst_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    frame = pd.read_csv(labels_path, dtype={"id": str, "Biosample": str}).set_index("id")
    biosample = frame.pop("Biosample")
    mlst = pd.read_csv(mlst_path, sep="\t", header=None, dtype=str)
    mlst.index = mlst[0].str.rsplit("/", n=1).str[-1].str.replace(".fasta", "", regex=False)
    alleles = mlst[list(range(3, 10))].apply(lambda col: col.str.extract(r"\((\d+)\)$")[0])
    alleles.insert(0, "st", mlst[2])
    return frame.astype(float), biosample, alleles.loc[frame.index]


def deduplicate(labels: pd.DataFrame, biosample: pd.Series) -> list[str]:
    """Keep one genome per BioSample: most phenotypes, then smallest ID."""
    order = pd.DataFrame({"n": labels.notna().sum(axis=1), "bs": biosample, "isolate": labels.index})
    order = order.reset_index(drop=True).sort_values(["bs", "n", "isolate"], ascending=[True, False, True])
    return sorted(order.drop_duplicates("bs")["isolate"])


def clonal_groups(alleles: pd.DataFrame) -> pd.Series:
    """Greedy one-hop star clustering: large STs become centres; single-locus variants join them."""
    known = alleles[alleles["st"].str.fullmatch(r"\d+") & alleles.iloc[:, 1:].notna().all(axis=1)]
    profiles = known.drop_duplicates("st").set_index("st").iloc[:, :7]
    sizes = known["st"].value_counts()
    centre_of: dict[str, str] = {}
    for st in sorted(sizes.index, key=lambda s: (-sizes[s], int(s))):
        diffs = (profiles.loc[list(dict.fromkeys(centre_of.values()))] != profiles.loc[st]).sum(axis=1) \
            if centre_of else pd.Series(dtype=int)
        near = [c for c in diffs.index if diffs[c] == 1]
        centre_of[st] = max(near, key=lambda c: (sizes[c], -int(c))) if near else st
    groups = {}
    for isolate, row in alleles.iterrows():
        if row["st"] in centre_of:
            groups[isolate] = f"CG{centre_of[row['st']]}"
            continue
        # unknown ST: join the clonal group if all STs matching >= 6/7 known alleles share it
        observed = row.iloc[1:].notna()
        matches = (profiles.loc[:, observed.values] == row.iloc[1:][observed].values).sum(axis=1)
        centres = {centre_of[st] for st in matches.index[matches >= 6]}
        groups[isolate] = f"CG{centres.pop()}" if len(centres) == 1 else f"novel:{isolate}"
    return pd.Series(groups, name="group")


def balanced_folds(groups: pd.Series, labels: pd.DataFrame, k: int, rng: np.random.Generator,
                   forbid_fold0: set[str] = frozenset()) -> pd.Series:
    """Greedy group k-fold balancing isolate count and per-drug R/S counts (largest groups first)."""
    lab = labels.loc[groups.index]
    counts = pd.concat([pd.Series(1.0, index=lab.index, name="n"), (lab == 1).astype(float).add_suffix(":R"),
                        (lab == 0).astype(float).add_suffix(":S")], axis=1).groupby(groups).sum()
    target = counts.sum() / k
    target[target == 0] = 1.0
    names = counts.index.to_numpy().copy()
    rng.shuffle(names)
    names = sorted(names, key=lambda g: -counts.loc[g, "n"])  # stable: random order within equal sizes
    totals = np.zeros((k, counts.shape[1]))
    assignment = {}
    for name in names:
        vector = counts.loc[name].to_numpy()
        scale = target.to_numpy()
        cost = (((totals + vector) / scale) ** 2).sum(axis=1) - ((totals / scale) ** 2).sum(axis=1)
        if name in forbid_fold0:
            cost[0] = np.inf
        fold = int(np.argmin(cost))
        totals[fold] += vector
        assignment[name] = fold
    return groups.map(assignment).astype(int)


def minority_ok(labels: pd.DataFrame, members: pd.Index, minimum: int) -> bool:
    sub = labels.loc[members]
    return bool(((sub == 1).sum().clip(upper=(sub == 0).sum()) >= minimum).all())


def write_partition(directory: Path, frame: pd.DataFrame, labels: pd.DataFrame, drugs: list[str]) -> list[dict]:
    """Write partition.tsv plus per-drug train/val/test ids; return manifest rows.

    Single-drug partitions (tier 1) keep their id files next to partition.tsv.
    """
    directory.mkdir(parents=True, exist_ok=True)
    frame = frame.sort_index()
    frame.rename_axis("isolate_id").reset_index()[["isolate_id", "group", "outer", "inner_fold"]].to_csv(
        directory / "partition.tsv", sep="\t", index=False)
    rows = []
    for drug in drugs:
        labelled = labels.index[labels[drug].notna()]
        sets = {"train": frame.index[(frame["outer"] == "train")].intersection(labelled),
                "val": frame.index[(frame["outer"] == "train") & (frame["inner_fold"] == 0)].intersection(labelled),
                "test": frame.index[frame["outer"] == "test"].intersection(labelled)}
        target = directory if len(drugs) == 1 else directory / drug
        target.mkdir(exist_ok=True)
        row: dict = {"drug": drug}
        for part, ids in sets.items():
            path = target / f"{part}.ids"
            path.write_text("".join(f"{i}\n" for i in sorted(ids)))
            row.update({f"n_{part}": len(ids), f"R_{part}": int((labels.loc[ids, drug] == 1).sum()),
                        f"S_{part}": int((labels.loc[ids, drug] == 0).sum()), f"sha256_{part}": sha256(path)})
        rows.append(row)
    return rows


def lineage_overlap(frame: pd.DataFrame, alleles: pd.DataFrame) -> tuple[float, float]:
    """Share of test isolates whose ST, or a single-locus variant of it, occurs in training."""
    train, test = frame.index[frame["outer"] == "train"], frame.index[frame["outer"] == "test"]
    profiles = alleles.iloc[:, 1:].astype(float)
    train_profiles = profiles.loc[train].dropna().drop_duplicates().to_numpy()
    seen_st = set(alleles.loc[train, "st"])
    same = np.mean([alleles.at[i, "st"] in seen_st and alleles.at[i, "st"] != "-" for i in test])
    slv = []
    for i in test:
        p = profiles.loc[i].to_numpy()
        slv.append(bool(np.isfinite(p).all() and ((train_profiles != p).sum(axis=1) <= 1).any()))
    return float(same), float(np.mean(slv))


def make_grouped_tier(name: str, groups: pd.Series, labels: pd.DataFrame, alleles: pd.DataFrame,
                      out: Path, base_seed: int) -> list[dict]:
    drugs, rows = list(labels.columns), []
    big = set(groups.value_counts()[lambda s: s > BIG_GROUP_FRACTION * len(groups)].index)
    for repeat in range(REPEATS):
        for attempt in range(50):
            outer = balanced_folds(groups, labels, FOLDS, np.random.default_rng(base_seed + 100 * repeat + attempt))
            if all(minority_ok(labels, outer.index[outer == f], MIN_MINORITY_TEST) for f in range(FOLDS)):
                break
        else:
            raise RuntimeError(f"{name}: no balanced outer split for repeat {repeat}")
        for fold in range(FOLDS):
            seed = repeat * FOLDS + fold
            train = outer.index[outer != fold]
            for inner_attempt in range(50):
                rng = np.random.default_rng(base_seed + 1000 + 10 * seed + inner_attempt)
                inner = balanced_folds(groups.loc[train], labels, FOLDS, rng, forbid_fold0=big)
                if minority_ok(labels, inner.index[inner == 0], MIN_MINORITY_VAL):
                    break
            else:
                raise RuntimeError(f"{name}: no valid inner folds for seed {seed}")
            frame = pd.DataFrame({"group": groups, "outer": np.where(outer == fold, "test", "train")})
            frame["inner_fold"] = inner.reindex(frame.index).fillna(-1).astype(int)
            same, slv = lineage_overlap(frame, alleles)
            for row in write_partition(out / name / f"seed{seed}", frame, labels, drugs):
                rows.append({"scheme": name, "seed": seed, "outer_attempt": attempt, "inner_attempt": inner_attempt,
                             "test_st_seen_in_train": same, "test_slv_in_train": slv, **row})
    return rows


def make_amrgnn_tier(labels: pd.DataFrame, biosample: pd.Series, alleles: pd.DataFrame, root: Path,
                     out: Path) -> list[dict]:
    """Tier 1: per target drug, test = AMR-GNN test; train pool = everything else minus test BioSample twins."""
    rows = []
    for seed in range(SEEDS):
        relative = "exact_seed0_64_16_20" if seed == 0 else f"fixed_test_multiseed/seed{seed}"
        source = root / relative / "amrgnn/splits"
        for drug in labels.columns:
            test, val = read_ids(source / drug / "test.ids"), set(read_ids(source / drug / "val.ids"))
            twins = set(biosample.index[biosample.isin(biosample.loc[test])]) - set(test)
            pool = labels.index.difference(test).difference(sorted(twins))
            val_twins = set(biosample.index[biosample.isin(biosample.loc[sorted(val)])]) - val
            val |= val_twins & set(pool)  # keep BioSample twins of validation isolates in fold 0 too
            rest = pool.difference(sorted(val))
            inner = balanced_folds(biosample.loc[rest], labels, FOLDS - 1, np.random.default_rng(7000 + seed)) + 1
            frame = pd.DataFrame({"group": biosample, "outer": "excluded", "inner_fold": -1})
            frame.loc[pool, "outer"] = "train"
            frame.loc[test, "outer"] = "test"
            frame.loc[sorted(val), "inner_fold"] = 0
            frame.loc[inner.index, "inner_fold"] = inner
            others = labels.index[labels[drug].notna()].difference(test)
            crossings = int(biosample.loc[test].isin(biosample.loc[others]).sum())
            same, slv = lineage_overlap(frame, alleles)
            for row in write_partition(out / "amrgnn_fixed" / f"seed{seed}" / drug, frame, labels[[drug]], [drug]):
                copied = read_ids(out / "amrgnn_fixed" / f"seed{seed}" / drug / "test.ids")
                if copied != sorted(test):
                    raise RuntimeError(f"Tier-1 test ids changed for {drug}, seed {seed}")
                rows.append({"scheme": "amrgnn_fixed", "seed": seed, "biosample_crossings_single_drug": crossings,
                             "n_twins_excluded": len(twins), "test_st_seen_in_train": same,
                             "test_slv_in_train": slv, **row})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=KP / "phenotype/ast_labels.csv")
    parser.add_argument("--mlst", type=Path, default=KP / "exact_seed0_64_16_20/amrgnn/mlst/mlst_results.tsv")
    parser.add_argument("--amrgnn-root", type=Path, default=KP)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty {args.out}")
    labels, biosample, alleles = load_inputs(args.labels, args.mlst)
    kept = deduplicate(labels, biosample)
    cg = clonal_groups(alleles).loc[kept]
    rows = make_amrgnn_tier(labels, biosample, alleles, args.amrgnn_root, args.out)
    rows += make_grouped_tier("random_grouped", pd.Series(kept, index=kept, name="group"), labels.loc[kept],
                              alleles, args.out, args.seed)
    rows += make_grouped_tier("cg_blocked", cg, labels.loc[kept], alleles, args.out, args.seed + 1)
    pd.DataFrame(rows).to_csv(args.out / "manifest.tsv", sep="\t", index=False)
    cg.rename_axis("isolate_id").to_csv(args.out / "clonal_groups.tsv", sep="\t")
    script = Path(__file__).resolve()
    shutil.copy2(script, args.out / script.name)
    (args.out / "split_protocol.json").write_text(json.dumps({
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"), "script_sha256": sha256(script),
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "inputs_sha256": {"labels": sha256(args.labels), "mlst": sha256(args.mlst)},
        "n_isolates": len(labels), "n_deduplicated": len(kept), "n_clonal_groups": int(cg.nunique()),
        "largest_groups": {k: int(v) for k, v in cg.value_counts().head(5).items()},
        "folds": FOLDS, "repeats": REPEATS, "min_minority_test": MIN_MINORITY_TEST,
        "min_minority_val": MIN_MINORITY_VAL, "big_group_fraction": BIG_GROUP_FRACTION,
    }, indent=2))
    print(f"Wrote {len(rows)} manifest rows to {args.out}")


if __name__ == "__main__":
    main()
