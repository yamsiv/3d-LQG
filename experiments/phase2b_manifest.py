"""Final Task2B provenance, archival preservation and local-link verification."""
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from src.provenance import environment_manifest


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=Path("results/phase2b/final_manifest.json")
    if output.exists():raise FileExistsError(output)
    original=json.loads(Path("results/final_manifest.json").read_text())
    preservation=[]
    for filename,entry in original["datasets"].items():
        path=Path(filename)
        current=digest(path)
        if current!=entry["sha256"]:raise ValueError(f"Original data changed: {path}")
        preservation.append(dict(file=filename,sha256=current,unchanged=True))
    documents=[Path(p) for p in ["README.md","Agents_Info/RESEARCH_LOG.md",
        "results/RESEARCH_REPORT_PHASE_2B.md","research-notes/questions.md"]]
    missing=[]
    for path in documents:
        for link in re.findall(r"\]\(([^)]+)\)",path.read_text()):
            if link.startswith(("http:","https:")):continue
            if not (path.parent/link.split("#")[0]).exists():missing.append((str(path),link))
    if missing:raise ValueError(f"Missing report links: {missing}")
    xml=ET.parse("results/phase2b/final_tests.xml")
    suites=xml.findall(".//testsuite")
    tests={key:sum(int(s.get(key,0)) for s in suites) for key in ["tests","errors","failures","skipped"]}
    if tests["errors"] or tests["failures"]:raise ValueError("Final tests did not pass")
    files=list(Path("results/phase2b").rglob("*"))+documents+list(Path("docs").glob("*PHASE2B.md"))
    artifacts={str(p):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files
               if p.is_file() and p!=output}
    result=dict(provenance=environment_manifest(),task="Research Task2B",tests=tests,
        original_dataset_preservation=preservation,local_report_links_valid=True,
        artifacts=artifacts,decision="stop_at_general_sampler_validation_obstacle_no_large_covariance_sweep",
        sampler_status={"exact_I3":"validated", "Weyl":"exact_grid_values_polygonal_minima_provisional",
            "analytic_Doob":"provisional_finite_step_bias_and_joint_minimum_error_budget",
            "general_FEM_Doob":"unvalidated_nonharmonic_boundary_approximation",
            "general_MCMC":"unvalidated_finite_grid_unbiased_alternative"},
        next_phase_started=False)
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(tests=tests,unchanged_original_datasets=len(preservation),
                          indexed_artifacts=len(artifacts),output=str(output))))


if __name__=="__main__":main()
