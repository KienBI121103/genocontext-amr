from __future__ import annotations

import logging
import subprocess
from pathlib import Path

LOG = logging.getLogger(__name__)


def sample_name(genome_path: Path) -> str:
    name = genome_path.name
    if name.endswith(".gz"):
        name = name[:-3]
    for suffix in (".fasta", ".fna", ".fa", ".fas"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"Unsupported genome FASTA name: {genome_path}")


def validate_gff3(path: Path) -> None:
    """Minimal completeness check; the full parser checks every feature row."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty GFF3: {path}")
    header = False
    cds = False
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("##gff-version 3"):
                header = True
            if line.startswith("##FASTA"):
                break
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9:
                raise ValueError(f"Malformed GFF3 row at {path}:{line_number}")
            if parts[2] == "CDS":
                cds = True
    if not header or not cds:
        raise ValueError(f"GFF3 requires a version-3 header and at least one CDS: {path}")


def ensure_bakta_annotation(
    genome_path: str | Path,
    output_dir: str | Path,
    bakta_db: str | Path,
    threads: int = 8,
) -> Path:
    """Reuse an existing GFF3; run Bakta only when it is absent."""
    genome_path = Path(genome_path)
    output_dir = Path(output_dir)
    gff = output_dir / f"{sample_name(genome_path)}.gff3"
    if gff.exists():
        validate_gff3(gff)
        LOG.info("Reusing Bakta GFF3: %s", gff)
        return gff
    if not genome_path.is_file():
        raise FileNotFoundError(f"Genome FASTA required for Bakta: {genome_path}")
    db = Path(bakta_db)
    if not db.is_dir():
        raise FileNotFoundError(f"Bakta database not found: {db}")
    if threads < 1:
        raise ValueError("threads must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "bakta", "--output", str(output_dir), "--prefix", sample_name(genome_path),
        "--db", str(db), "--threads", str(threads), "--skip-plot", "--force",
    ]
    command.append(str(genome_path))
    LOG.info("Running Bakta for %s", genome_path)
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as exc:
        raise RuntimeError(f"Cannot start Bakta for {genome_path}: {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(f"Bakta failed for {genome_path} (exit {result.returncode}): {result.stderr[-2000:]}")
    validate_gff3(gff)
    return gff
