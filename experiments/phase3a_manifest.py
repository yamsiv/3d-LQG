"""Final provenance, raw-checkpoint validation and preservation for Task 3A."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
from src.provenance import environment_manifest


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path("results/phase3a");output=root/"final_manifest.json"
    if output.exists():raise FileExistsError(output)
    mutable={"README.md","Agents_Info/RESEARCH_LOG.md","research-notes/questions.md"}
    old=json.loads(Path("results/phase2c/final_manifest.json").read_text())
    old_sources=old["provenance"]["source_sha256"]
    old_artifacts={k:v["sha256"]for k,v in old["artifacts"].items() if k not in mutable}
    for filename,expected in {**old_sources,**old_artifacts}.items():
        if digest(filename)!=expected:raise ValueError(f"Prior artifact changed: {filename}")
    original=json.loads(Path("results/final_manifest.json").read_text())
    for filename,value in original["datasets"].items():
        if digest(filename)!=value["sha256"]:raise ValueError(f"Original data changed: {filename}")
    suites=ET.parse(root/"final_tests.xml").findall(".//testsuite")
    tests={k:sum(int(s.get(k,0)) for s in suites)for k in ["tests","errors","failures","skipped"]}
    if tests["tests"]<145 or tests["errors"] or tests["failures"]:raise ValueError(tests)
    stage_counts={};seeds={}
    for folder,expected in [("discovery",144),("confirmation",144),("fixed_field_mesh",8),
            ("heat_regularizer",8),("open_cube",8),("flat_exact",24),("smooth_control",8)]:
        files=sorted((root/folder).glob("metric*.npz"))
        if len(files)!=expected:raise ValueError(f"Incomplete stage {folder}: {len(files)}/{expected}")
        stage_counts[folder]=len(files);seeds[folder]=set()
        for path in files:
            with np.load(path) as a:
                header=json.loads(str(a["header"]))
                if "field_seed" in header:seeds[folder].add(header["field_seed"])
                c=a["cdfs"];m=a["matrices"]
                if np.any(np.diff(c,axis=-1)<-2e-7) or not np.allclose(c[:,:,-1],1,atol=2e-7):raise ValueError(f"Invalid CDF {path}")
                if not np.allclose(m,np.swapaxes(m,-1,-2),atol=1e-10):raise ValueError(f"Asymmetric metric {path}")
                if not np.allclose(np.diagonal(m,axis1=-2,axis2=-1),0):raise ValueError(f"Nonzero metric diagonal {path}")
                if np.any(m[:,:,:,None]>m[:,:,None,:]+m[:,None,:,:]+1e-9):raise ValueError(f"Triangle failure {path}")
    if seeds["discovery"]&seeds["confirmation"]:raise ValueError("Discovery/confirmation seed collision")
    graph_files=list((root/"graphs").glob("graph*.npz"))
    if len(graph_files)!=288:raise ValueError("Incomplete graph ensemble")
    crc_count=0
    for path in root.rglob("*.npz"):
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:raise ValueError(f"Bad archive {path}")
        crc_count+=1

    provenance=environment_manifest()
    snapshot=root/"source_snapshots";snapshot.mkdir(exist_ok=True)
    source_index={}
    for name,sha in provenance["source_sha256"].items():
        if not name.endswith(".py"):continue
        destination=snapshot/(sha+".py")
        if not destination.exists():destination.write_bytes(Path(name).read_bytes())
        if digest(destination)!=sha:raise ValueError("Source snapshot corrupted")
        source_index[name]={"sha256":sha,"snapshot":str(destination)}
    index_path=snapshot/"final_source_index.json"
    if index_path.exists():raise FileExistsError(index_path)
    index_path.write_text(json.dumps(source_index,indent=2)+"\n")
    recovered=json.loads((snapshot/"recovered_version_index.json").read_text())
    for row in recovered:
        if digest(row["frozen_file"])!=row["sha256"]:raise ValueError("Recovered source hash failed")
    documents=[Path("README.md"),Path("Agents_Info/RESEARCH_LOG.md"),Path("research-notes/questions.md"),
        Path("results/RESEARCH_REPORT_PHASE_3A.md"),root/"STATUS.md"]+list(Path("docs").glob("*PHASE3A.md"))
    missing=[]
    for p in documents:
        for link in re.findall(r"\]\(([^)]+)\)",p.read_text()):
            if link.startswith(("http:","https:")):continue
            target=(p.parent/link.split("#")[0]).resolve()
            if not target.exists() and target!=output.resolve():missing.append((str(p),link))
    if missing:raise ValueError(f"Broken links: {missing}")
    artifacts=sorted(set(p for p in list(root.rglob("*"))+documents if p.is_file() and p!=output))
    manifest={"task":"Research Task 3A","decision":"Outcome B: partial or ambiguous correspondence",
        "provenance":provenance,"tests":tests,"prior_phase2c_sources_unchanged":len(old_sources),
        "prior_phase2c_artifacts_unchanged":len(old_artifacts),"original_datasets_unchanged":len(original["datasets"]),
        "metric_checkpoint_counts":stage_counts,"graph_checkpoints":len(graph_files),
        "independent_discovery_field_seeds":len(seeds["discovery"]),"independent_confirmation_field_seeds":len(seeds["confirmation"]),
        "independent_seed_sets_disjoint":True,"npz_crc_checks":crc_count,"local_document_links_valid":True,
        "source_snapshots":source_index,"source_version_note":"Producer versions for discovery/graphs/confirmation recovered and verified against original metadata hashes; unrelated validation-file hash changes do not change those producers.",
        "original_graph_generators_modified":False,"general_covariance_sweep":False,"new_major_phase_started":False,
        "artifacts":{str(p):{"sha256":digest(p),"bytes":p.stat().st_size}for p in artifacts}}
    output.write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"output":str(output),"tests":tests,"artifacts":len(artifacts),"npz_checks":crc_count,
        "prior_sources_preserved":len(old_sources),"prior_artifacts_preserved":len(old_artifacts)}))


if __name__=="__main__":main()
