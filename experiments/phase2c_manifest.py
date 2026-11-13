"""Verify the Task 2C handoff without running or replacing simulations.

Run once after the final tests. Output refuses overwriting; earlier manifests
remain independent historical records. Hashes are integrity checks, not a
certificate of mathematical correctness.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

from src.provenance import environment_manifest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unchanged(entries):
    checked = {}
    for filename, expected in entries.items():
        actual = digest(filename)
        if actual != expected:
            raise ValueError(f"Archived artifact changed: {filename}")
        checked[filename] = actual
    return checked


def main():
    output = Path("results/phase2c/final_manifest.json")
    if output.exists():
        raise FileExistsError(output)
    previous = json.loads(Path("results/phase2b/final_manifest.json").read_text())
    first = json.loads(Path("results/final_manifest.json").read_text())
    recovery = json.loads(Path("results/phase2c/resumption_audit.json").read_text())
    mutable = {"README.md", "Agents_Info/RESEARCH_LOG.md", "research-notes/questions.md"}
    preserved = {
        "phase2b_sources": unchanged(previous["provenance"]["source_sha256"]),
        "phase2b_artifacts": unchanged({
            k: v["sha256"] for k, v in previous["artifacts"].items() if k not in mutable
        }),
        "task1_datasets": unchanged({k: v["sha256"] for k, v in first["datasets"].items()}),
        "recovered_phase2c_outputs": unchanged({
            k: v["sha256"] for k, v in recovery["recovered_artifacts"].items()
            if k not in recovery["mutable_handoff_files"]
        }),
    }

    root = Path("results/phase2c")
    jsonl = {}
    for path in sorted(root.rglob("*.json")):
        json.loads(path.read_text())
    for path in sorted(root.rglob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        jsonl[str(path)] = len(rows)
    for name, entry in json.loads((root / "paths/completed_experiment_inventory.json").read_text()).items():
        filename = str(root / "paths" / name / "records.jsonl")
        if jsonl[filename] != entry["independent_checkpoint_rows"]:
            raise ValueError(f"Incomplete checkpoint: {filename}")
    archives = {}
    for path in sorted(root.rglob("*.npz")):
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            if bad_member:
                raise ValueError(f"Corrupt archive: {path}: {bad_member}")
            archives[str(path)] = {"members": len(archive.namelist()), "crc_ok": True}

    documents = [Path(p) for p in [
        "README.md", "Agents_Info/RESEARCH_LOG.md", "research-notes/questions.md",
        "results/RESEARCH_REPORT_PHASE_2C.md", "results/phase2c/STATUS.md",
    ]] + sorted(Path("docs").glob("*PHASE2C.md"))
    missing = []
    for path in documents:
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if link.startswith(("http:", "https:")):
                continue
            target = (path.parent / link.split("#")[0]).resolve()
            if not target.exists() and target != output.resolve():
                missing.append((str(path), link))
    if missing:
        raise ValueError(f"Missing document links: {missing}")

    test_file = root / "test_results_resumed.xml"
    suites = ET.parse(test_file).findall(".//testsuite")
    tests = {key: sum(int(s.get(key, 0)) for s in suites)
             for key in ("tests", "errors", "failures", "skipped")}
    if tests["tests"] < 121 or tests["errors"] or tests["failures"]:
        raise ValueError(f"Final suite incomplete or unsuccessful: {tests}")

    files = sorted(set(p for p in list(root.rglob("*")) + documents
                       if p.is_file() and p != output))
    result = {
        "task": "Research Task 2C",
        "decision": "Outcome C: complete general sampler remains unreliable",
        "provenance": environment_manifest(),
        "tests": tests,
        "test_file": str(test_file),
        "preservation_sha256": preserved,
        "intentionally_updated_handoff_files": sorted(mutable),
        "jsonl_rows": jsonl,
        "npz_integrity": archives,
        "local_document_links_valid": True,
        "resumed_expensive_experiments": False,
        "generic_graphs_generated": False,
        "new_dimension_fit": False,
        "field_gmc_lfpp_started": False,
        "sampler_status": {
            "I3_existing_exact_minima": "validated",
            "A2_matrix_bridge": "exact grid law; approximate joint continuous minima",
            "A3_matrix_bridge": "validated original/biased grid laws; approximate joint continuous minima",
            "improved_analytic_Doob": "provisional: full-path/minimum accuracy targets unestablished",
            "generic_angular_modes": "positive-formula and cross-method convergence evidence; no uniform path certificate",
            "generic_complete_sampler": "unvalidated; sampling gate closed",
        },
        "artifacts": {str(p): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in files},
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "tests": tests,
                      "preserved": {k: len(v) for k, v in preserved.items()},
                      "indexed_artifacts": len(files)}))


if __name__ == "__main__":
    main()
