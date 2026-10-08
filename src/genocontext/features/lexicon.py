"""Curated knowledge: drug classes, mechanisms, AMR naming rules and mobile-element detection."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from genocontext.data.gff import Gene

_MOBILE_PRODUCT = re.compile(r"transposase|integrase|resolvase|recombinase", re.I)
_NOT_MOBILE = re.compile(r"\bXer[CD]\b|\bRecA\b|site-specific tyrosine recombinase XerD", re.I)
_BLA_ALLELE = re.compile(r"-\d+[A-Z]?$")
_ALLELE_DIGITS = re.compile(r"(?<=[A-Z])\d+$")
_CR_VARIANT = re.compile(r"AAC\(6'\)-Ib-cr\d*")


@dataclass(frozen=True)
class AmrRule:
    mechanisms: tuple[str, ...]
    gene: re.Pattern[str] | None
    product: re.Pattern[str] | None

    def matches(self, gene: str, product: str) -> bool:
        return bool((self.gene and self.gene.search(gene)) or (self.product and self.product.search(product)))


@dataclass(frozen=True)
class Target:
    name: str
    mechanisms: tuple[str, ...]
    residues: dict[int, str]
    indels: bool


@dataclass(frozen=True)
class Knowledge:
    drugs: dict[str, tuple[str, ...]]
    mechanisms: dict[str, tuple[str, ...]]
    rules: tuple[AmrRule, ...]
    targets: dict[str, Target]
    sha256: str

    @property
    def mechanism_names(self) -> tuple[str, ...]:
        return tuple(self.mechanisms)

    @property
    def drug_classes(self) -> tuple[str, ...]:
        return tuple(sorted({c for classes in self.drugs.values() for c in classes}))

    def drug_features(self, drugs: tuple[str, ...]) -> NDArray[np.float32]:
        """Multi-hot drug-class matrix, shape (n_drugs, n_classes)."""
        classes = self.drug_classes
        return np.array([[c in self.drugs[d] for c in classes] for d in drugs], dtype=np.float32)

    def prior_mask(self, drugs: tuple[str, ...]) -> NDArray[np.float32]:
        """1 where a mechanism is known to affect a drug's class, shape (n_mechanisms, n_drugs)."""
        return np.array([[bool(set(affects) & set(self.drugs[d])) for d in drugs]
                         for affects in self.mechanisms.values()], dtype=np.float32)

    def amr_mechanisms(self, gene: str, product: str) -> tuple[str, ...]:
        for rule in self.rules:
            if rule.matches(gene, product):
                return rule.mechanisms
        return ("other",)


def _pattern(text: str | None) -> re.Pattern[str] | None:
    return re.compile(text, re.I) if text else None


def load_knowledge(path: str | Path) -> Knowledge:
    raw_text = Path(path).read_text(encoding="utf-8")
    raw: dict[str, Any] = yaml.safe_load(raw_text)
    mechanisms = {name: tuple(affects) for name, affects in raw["mechanisms"].items()}
    rules = tuple(AmrRule(tuple(r["mechanisms"]), _pattern(r.get("gene")), _pattern(r.get("product")))
                  for r in raw["amr_rules"])
    targets = {name: Target(name, tuple(spec["mechanisms"]), {int(k): v for k, v in spec.get("residues", {}).items()},
                            bool(spec.get("indels", False))) for name, spec in raw["targets"].items()}
    used = {m for r in rules for m in r.mechanisms} | {m for t in targets.values() for m in t.mechanisms}
    if not used <= set(mechanisms):
        raise ValueError(f"Undefined mechanisms in knowledge file: {sorted(used - set(mechanisms))}")
    return Knowledge({d: tuple(c) for d, c in raw["drugs"].items()}, mechanisms, rules, targets,
                     hashlib.sha256(raw_text.encode()).hexdigest())


def amr_names(gene: Gene) -> tuple[str, str] | None:
    """Allele and family names for an AMRFinderPlus-tagged CDS (Bakta ``NCBIProtein:`` xref), else None."""
    if gene.xref("NCBIProtein:") is None or not gene.gene:
        return None
    allele = gene.gene
    variant = _CR_VARIANT.search(gene.product or "")
    if variant:  # Bakta names the gene aac(6')-Ib; only the product carries the -cr variant
        allele = "aac(6')-Ib-cr"
    family = _BLA_ALLELE.sub("", allele) if allele.startswith("bla") else _ALLELE_DIGITS.sub("", allele)
    return allele, family


def mobile_name(gene: Gene) -> str | None:
    """Mobile-element token for IS elements, transposases, integrases and recombinases."""
    insertion = gene.xref("IS:")
    if insertion:
        return insertion
    if gene.gene and gene.gene.lower().startswith("inti"):
        return gene.gene
    product = gene.product or ""
    match = _MOBILE_PRODUCT.search(product)
    if match and not _NOT_MOBILE.search(product):
        return match.group(0).lower()
    return None


def family_name(gene: Gene) -> str | None:
    """Gene-family identity: UniRef90, then UniRef50, then the exact gene name."""
    for prefix in ("UniRef:UniRef90_", "UniRef:UniRef50_"):
        value = gene.xref(prefix)
        if value:
            return prefix.split(":")[1] + value
    return f"gene:{gene.gene}" if gene.gene else None
