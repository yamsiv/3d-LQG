"""Final saved-artifact, preservation and completeness audit for Phase 3B."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from src.provenance import environment_manifest


ROOT=Path("results/phase3b")


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric_check(path, arrays):
    c=np.asarray(arrays["cdfs"])
    if np.any(~np.isfinite(c)) or np.any(np.diff(c,axis=-1)<-2e-7) or not np.allclose(c[...,-1],1,atol=2e-7):
        raise ValueError(f"Invalid CDF: {path}")
    key="matrices" if "matrices" in arrays else "matrix"
    m=np.asarray(arrays[key]);m=m.reshape((-1,m.shape[-1],m.shape[-1]))
    if np.any(~np.isfinite(m)) or np.any(m<0) or not np.allclose(m,m.swapaxes(-1,-2)) or not np.allclose(np.diagonal(m,axis1=-2,axis2=-1),0):
        raise ValueError(f"Invalid distance matrix: {path}")
    if np.any(m[:,:,:,None]>m[:,:,None,:]+m[:,None,:,:]+1e-9):
        raise ValueError(f"Triangle inequality: {path}")


def main():
    output=ROOT/"final_manifest.json"
    if output.exists():raise FileExistsError(output)
    mutable={"README.md","Agents_Info/RESEARCH_LOG.md","research-notes/questions.md"}
    old=json.loads(Path("results/phase3a/final_manifest.json").read_text())
    old_sources=old["provenance"]["source_sha256"]
    old_artifacts={p:a["sha256"] for p,a in old["artifacts"].items() if p not in mutable}
    for path,expected in {**old_sources,**old_artifacts}.items():
        if digest(path)!=expected:raise ValueError(f"Prior artifact changed: {path}")
    original=json.loads(Path("results/final_manifest.json").read_text())
    for path,value in original["datasets"].items():
        if digest(path)!=value["sha256"]:raise ValueError(f"Original dataset changed: {path}")
    suites=ET.parse(ROOT/"final_tests.xml").findall(".//testsuite")
    tests={k:sum(int(s.get(k,0)) for s in suites) for k in ["tests","errors","failures","skipped"]}
    if tests["tests"]<173 or tests["errors"] or tests["failures"]:raise ValueError(tests)
    graph_counts={};graph_paths={};graph_seeds={};runtimes={};samplers={}
    expected_graphs={"screen_graphs":336,"adaptive_coarse_graphs":36,"intermediate_graphs":144,
                     "birect_extension_graphs":16,"holdout_graphs":64,"anchor_controls":36}
    for stage,expected in expected_graphs.items():
        paths=sorted((ROOT/stage).glob("C*/graph_*.npz"))
        if len(paths)!=expected:raise ValueError((stage,len(paths),expected))
        unique={};graph_seeds[stage]=set()
        for path in paths:
            with np.load(path) as arrays:
                h=json.loads(str(arrays["header"]));metric_check(path,arrays)
                v=h["validation"]
                if not v["connected"] or not v["hamiltonian_cycle"] or v["coordinate_diagonal_counts"]!=[h["n"]-3]*3:
                    raise ValueError(f"Graph invariant: {path}")
                identity=(h["id"],h["seed"]);unique[identity]=h
                graph_seeds[stage].add(h["seed"])
        graph_counts[stage]=len(paths);graph_paths[stage]=len(unique)
        runtimes[stage]=sum(h["seconds"] for h in unique.values())
        samplers[stage]=sorted({h["sampler_metadata"]["sampler"] for h in unique.values()})
    main_stages=[s for s in expected_graphs if s!="anchor_controls"]
    for i,a in enumerate(main_stages):
        for b in main_stages[i+1:]:
            if graph_seeds[a]&graph_seeds[b]:raise ValueError(f"Main graph stages share seeds: {a}, {b}")
    field_counts={};field_sources={};field_seeds={}
    expected_fields={"field_pilot":8,"screen_fields":88,"intermediate_fields":108,"holdout_fields":72,
                     "holdout_stencil_control":12,"holdout_smooth_control":16,"historical_candidate_control":12}
    for stage,expected in expected_fields.items():
        paths=sorted((ROOT/stage).glob("xi*/metric_*.npz"))
        if len(paths)!=expected:raise ValueError((stage,len(paths),expected))
        hashes={};seeds=set();seconds=0.
        for path in paths:
            with np.load(path) as arrays:
                h=json.loads(str(arrays["header"]));metric_check(path,arrays)
                hashes.setdefault(h["realization"],set()).add(h["source_field_sha256"])
                if h["field_seed"] is not None:seeds.add(h["field_seed"])
                source=Path(h["source_field"])
                if not source.exists() or digest(source)!=h["source_field_sha256"]:
                    raise ValueError(f"Cached field hash: {path}")
                seconds+=h["seconds"]
        if any(len(s)!=1 for s in hashes.values()):raise ValueError(f"Field identity changed across xi: {stage}")
        field_counts[stage]=len(paths);field_sources[stage]={str(k):next(iter(v)) for k,v in hashes.items()}
        field_seeds[stage]=seeds;runtimes[stage]=seconds
    discovery_hashes=set(field_sources["field_pilot"].values())|set(field_sources["screen_fields"].values())
    confirmation_hashes=set(field_sources["holdout_fields"].values())
    if len(discovery_hashes)!=12 or len(confirmation_hashes)!=12 or discovery_hashes&confirmation_hashes:
        raise ValueError("Discovery/heldout field identities")
    for stage in ["holdout_stencil_control","holdout_smooth_control","historical_candidate_control"]:
        if not set(field_sources[stage].values())<=confirmation_hashes:raise ValueError("Control regenerated fields")
    parsed={"json":0,"jsonl_rows":0,"npz_crc":0}
    for path in ROOT.rglob("*.json"):
        json.loads(path.read_text());parsed["json"]+=1
    for path in ROOT.rglob("*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():json.loads(line);parsed["jsonl_rows"]+=1
    for path in ROOT.rglob("*.npz"):
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:raise ValueError(f"Corrupt ZIP: {path}")
        parsed["npz_crc"]+=1
    provenance=environment_manifest();snapshots=ROOT/"source_snapshots";snapshots.mkdir(exist_ok=True)
    index={}
    for path,sha in provenance["source_sha256"].items():
        if not path.endswith(".py"):continue
        dest=snapshots/(sha+".py")
        if not dest.exists():dest.write_bytes(Path(path).read_bytes())
        if digest(dest)!=sha:raise ValueError(f"Snapshot hash: {dest}")
        index[path]={"sha256":sha,"snapshot":str(dest)}
    index_path=snapshots/"final_source_index.json"
    if index_path.exists():raise FileExistsError(index_path)
    index_path.write_text(json.dumps(index,indent=2)+"\n")
    documents=[Path("README.md"),Path("Agents_Info/RESEARCH_LOG.md"),Path("research-notes/questions.md"),
               Path("results/RESEARCH_REPORT_PHASE_3B.md"),ROOT/"STATUS.md"]+list(Path("docs").glob("*PHASE3B.md"))
    missing=[]
    for path in documents:
        for link in re.findall(r"\]\(([^)]+)\)",path.read_text()):
            if link.startswith(("http:","https:")):continue
            target=(path.parent/link.split("#")[0]).resolve()
            if not target.exists() and target!=output.resolve():missing.append((str(path),link))
    if missing:raise ValueError(f"Broken local links: {missing}")
    quality=json.loads((ROOT/"quality/audited/summary.json").read_text())
    artifacts=sorted(set(p for p in list(ROOT.rglob("*"))+documents if p.is_file() and p!=output))
    manifest={"task":"Research Task 3B","decision":"Partial or ambiguous correspondence; no unique covariance map identified",
        "provenance":provenance,"tests":tests,"prior_phase3a_sources_unchanged":len(old_sources),
        "prior_phase3a_artifacts_unchanged":len(old_artifacts),"original_datasets_unchanged":len(original["datasets"]),
        "graph_checkpoint_counts":graph_counts,"stage_qualified_physical_paths":graph_paths,
        "shared_seed_anchor_groups":len(quality["shared_seed_cross_stage_groups"]),
        "shared_seed_note":"18 exact/provisional anchor groups share RNG seeds across generators and are not pooled as independent; all main search/holdout graph-stage seed sets are disjoint",
        "metric_checkpoint_counts":field_counts,"field_source_hashes":field_sources,
        "independent_search_fields_reused_from_phase3a":12,"independent_new_holdout_fields":12,
        "discovery_holdout_field_hashes_disjoint":True,"controls_reuse_exact_cached_holdout_fields":True,
        "producer_seconds_excluding_model_setup":runtimes,"samplers":samplers,"parsed":parsed,
        "source_snapshots":index,"local_document_links_valid":True,
        "source_version_note":"Current complete source set frozen by SHA256; earlier executed producer versions are retained in per-stage snapshots and the shared source_snapshots directory; each stage records its original source hashes",
        "new_major_phase_started":False,"artifacts":{str(p):{"sha256":digest(p),"bytes":p.stat().st_size} for p in artifacts}}
    output.write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"output":str(output),"tests":tests,"artifacts":len(artifacts),"parsed":parsed,
        "prior_sources_preserved":len(old_sources),"prior_artifacts_preserved":len(old_artifacts),
        "producer_seconds":runtimes}))


if __name__=="__main__":main()
