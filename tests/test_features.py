from pathlib import Path

import numpy as np

from genocontext.config import FeatureConfig
from genocontext.data.gff import Gene, parse_gff3
from genocontext.features.lexicon import Knowledge, Target, amr_names, family_name, mobile_name
from genocontext.features.profile import Node, Profile, extract_profile, save_profile
from genocontext.features.space import FeatureSpace
from genocontext.features.store import ProfileStore
from genocontext.features.targets import TargetCaller
from tests.conftest import gff_line, write_annotation

REF = "MSTNPKLLVIGAGGLGSALAYDLARQGHEVTLVEKRDQ" + "W" * 10 + "SYGAMGDRTQWLLKEHGIEVLAPDQRF"


def gene(product: str = "x", gene_name: str | None = None, xrefs: tuple[str, ...] = ()) -> Gene:
    return Gene("c", "L1", 1, 100, "+", gene_name, product, xrefs, False)


def test_amr_naming_rules(knowledge: Knowledge) -> None:
    ctx = gene("class A beta-lactamase", "blaCTX-M-15", ("NCBIProtein:WP_1",))
    assert amr_names(ctx) == ("blaCTX-M-15", "blaCTX-M")
    assert amr_names(gene("x", "qnrB1", ("NCBIProtein:WP_1",))) == ("qnrB1", "qnrB")
    assert amr_names(gene("x", "sul1", ("NCBIProtein:WP_1",))) == ("sul1", "sul1")
    cr = gene("fluoroquinolone-acetylating aminoglycoside 6'-N-acetyltransferase AAC(6')-Ib-cr5", "aac(6')-Ib",
              ("NCBIProtein:WP_1",))
    assert amr_names(cr) == ("aac(6')-Ib-cr", "aac(6')-Ib-cr")
    assert amr_names(gene("sulfate permease", "sUL1")) is None  # no AMRFinderPlus tag
    assert knowledge.amr_mechanisms("blaKPC-3", "carbapenem-hydrolyzing class A beta-lactamase KPC-3") == (
        "carbapenemase",)
    assert knowledge.amr_mechanisms("aac(6')-Ib-cr", cr.product or "") == ("pmqr", "ame")
    assert knowledge.amr_mechanisms("fosA", "") == ("other",)


def test_mobile_and_family_names() -> None:
    assert mobile_name(gene("IS5 family transposase", xrefs=("IS:ISKpn26",))) == "ISKpn26"
    assert mobile_name(gene("site-specific tyrosine recombinase XerD")) is None
    assert mobile_name(gene("tyrosine-type recombinase/integrase")) == "recombinase"
    assert family_name(gene(xrefs=("UniRef:UniRef50_B", "UniRef:UniRef90_A"))) == "UniRef90_A"
    assert family_name(gene(gene_name="abc")) == "gene:abc"


def test_prior_mask_links_mechanisms_to_drug_classes(knowledge: Knowledge) -> None:
    drugs = ("imipenem", "ciprofloxacin", "trimethoprim_sulfamethoxazole")
    mask = dict(zip(knowledge.mechanism_names, knowledge.prior_mask(drugs), strict=True))
    assert mask["carbapenemase"].tolist() == [1, 0, 0]
    assert mask["fq_target"].tolist() == [0, 1, 0]
    assert mask["folate"].tolist() == [0, 0, 1]
    assert mask["other"].sum() == 0


def _caller(residues: dict[int, str] | None = None) -> TargetCaller:
    targets = {"T": Target("T", ("fq_target",), residues or {}, True)}
    return TargetCaller({"T": REF}, targets, FeatureConfig(contig_edge_bp=50))


def test_target_states_residues_and_indels(tmp_path: Path) -> None:
    mutant = REF[:5] + "I" + REF[6:20] + "GD" + REF[20:]  # residue 6 S->I and a GD insertion after position 20
    cases = {
        "intact": ([gff_line("c", 500, 760, "A")], {"A": mutant}),
        "truncated": ([gff_line("c", 500, 650, "A")], {"A": REF[30:]}),
        "is_disrupted": ([gff_line("c", 500, 650, "A"), gff_line("c", 660, 900, "IS", xrefs="IS:ISKpn6")],
                         {"A": REF[30:], "IS": "MAAAAAAAAAAAAAAAAAAAAA"}),
        "edge_unknown": ([gff_line("c", 5, 150, "A")], {"A": REF[30:]}),
        "absent": ([gff_line("c", 500, 650, "A")], {"A": "M" + "PQ" * 40}),
    }
    caller = _caller({6: "S"})
    for expected, (cds, proteins) in cases.items():
        gff, _ = write_annotation(tmp_path, expected, {"c": 5000}, cds, proteins)
        call = caller.call(parse_gff3(gff), proteins, {"IS"})[0]
        assert call.state == expected, expected
        if expected == "intact":
            assert "T_S6I" in call.variants and "T_ins20_GD" in call.variants


def _store(tmp_path: Path, knowledge: Knowledge, nodes_per_isolate: list[list[Node]]) -> ProfileStore:
    isolates = [f"i{r}" for r in range(len(nodes_per_isolate))]
    for isolate, nodes in zip(isolates, nodes_per_isolate, strict=True):
        save_profile(Profile(isolate, {"n_contigs": 1.0}, nodes), tmp_path / f"{isolate}.json.gz")
    return ProfileStore.build(tmp_path, isolates, knowledge.mechanism_names, threads=2)


def node(token: str, group: str = "fam", positions: list[list[int]] | None = None, state: str = "present",
         mech: str = "other") -> Node:
    return Node(token, group, state, [mech], [0, 0, 0], positions or [[0, 0]], ["L"])


def test_feature_space_is_fitted_on_training_rows_only(tmp_path: Path, knowledge: Knowledge) -> None:
    profiles = [[node("fam:A", positions=[[0, 1]]), node("fam:B", positions=[[0, 3]]),
                 node("amr:blaKPC-3", "amr", [[0, 2]], mech="carbapenemase"),
                 node("target:GyrA", "target", [[0, 9]], "truncated", "fq_target")] for _ in range(6)]
    for r in (0, 1, 2):  # fam:A present in half the isolates, fam:C only in a test isolate
        profiles[r] = [n for n in profiles[r] if n.token != "fam:A"]
    profiles[5].append(node("fam:C", positions=[[0, 4]]))
    store = _store(tmp_path, knowledge, profiles)
    config = FeatureConfig(min_prevalence=1, max_families=10)
    space = FeatureSpace.fit(store, np.arange(5), config)
    assert "fam:C" not in space.prevalence  # never seen in training rows
    graph = space.graph(store, 5, k=2)
    tokens = [space.blocks[t] for t in graph["token"]]
    assert "fam:C" not in tokens and "target:GyrA" in tokens
    edges = {(tokens[a], tokens[b]) for a, b in graph["edge_index"].T}
    assert ("fam:A", "amr:blaKPC-3") in edges and ("fam:A", "target:GyrA") not in edges
    mech = {tokens[n]: knowledge.mechanism_names[m] for n, m in zip(graph["pair_node"], graph["pair_mech"],
                                                                     strict=True)}
    assert mech["amr:blaKPC-3"] == "carbapenemase" and mech["target:GyrA"] == "fq_target"
    shuffled = space.graph(store, 5, k=2, shuffle_seed=1)
    assert np.array_equal(shuffled["token"], graph["token"])  # same nodes, possibly different edges
    assert np.array_equal(space.graph(store, 5, 2, 1)["edge_index"], shuffled["edge_index"])  # deterministic
    columns = space.column_names("genestate")
    assert "target:GyrA=truncated" in columns and "target:GyrA=intact" in columns
    assert space.matrix(store, np.array([5]))[0, columns.index("target:GyrA=truncated")] == 1


def test_extract_profile_end_to_end(tmp_path: Path, knowledge: Knowledge) -> None:
    gff, faa = write_annotation(tmp_path, "iso", {"c": 9000}, [
        gff_line("c", 1000, 1800, "K", gene="blaKPC-3", product="carbapenem-hydrolyzing class A beta-lactamase KPC-3",
                 xrefs="NCBIProtein:WP_1"),
        gff_line("c", 1900, 2900, "T", xrefs="IS:ISKpn6", product="transposase"),
        gff_line("c", 3000, 3300, "F", xrefs="UniRef:UniRef90_Q"),
    ], {"K": "MSTNPK", "T": "MAAAA", "F": "MQQQQ"})
    caller = TargetCaller({"GyrA": REF}, {"GyrA": Target("GyrA", ("fq_target",), {}, False)}, FeatureConfig())
    profile = extract_profile("iso", gff, faa, knowledge, caller, FeatureConfig())
    by_token = {n.token: n for n in profile.nodes}
    assert by_token["amr:blaKPC-3"].mechanisms == ["carbapenemase"]
    assert by_token["amr:blaKPC-3"].flags[2] == 1  # near a mobile element
    assert by_token["target:GyrA"].state == "absent"
    assert {"mge:ISKpn6", "fam:UniRef90_Q"} <= by_token.keys()
