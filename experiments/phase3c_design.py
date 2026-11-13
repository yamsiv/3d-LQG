"""Freeze the broad design and input-preservation baseline before simulation."""
from pathlib import Path
import hashlib
import json
from collections import Counter
from src.covariance_design_3c import broad_design
from src.provenance import environment_manifest


def main():
    root=Path('results/phase3c');folder=root/'covariance_design';folder.mkdir(parents=True,exist_ok=True)
    out=folder/'initial_design.json'
    if out.exists():raise FileExistsError(out)
    old=json.loads(Path('results/phase3b/extended_design.json').read_text())['configurations']
    specs=broad_design(old)
    data={'configurations':specs,'counts':dict(Counter(g for s in specs for g in s['coverage_families'])),
        'old_configurations':len(old),'new_configurations':len(specs)-len(old),
        'adaptive_slots_reserved':12,'seed':202611123011,'condition_limit':12.,'minimum_eigenvalue':.15,
        'design_only_not_a_claim_of_completed_simulation':True,'provenance':environment_manifest()}
    out.write_text(json.dumps(data,indent=2)+'\n')
    manifest=Path('results/phase3b/finalization_20261112/final_manifest.json')
    prior=json.loads(manifest.read_text());mismatches=[]
    for path,row in prior['artifacts'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=row['sha256']:mismatches.append(path)
    if mismatches:raise ValueError(f'Prior sealed artifacts changed: {mismatches}')
    (root/'recovery.json').write_text(json.dumps({'prior_manifest':str(manifest),
        'prior_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'prior_artifacts_verified':len(prior['artifacts']),'mismatches':mismatches,
        'prior_source_hashes':prior['provenance']['source_sha256'],
        'new_phase':'3C explicitly authorized; no prior dataset overwritten'},indent=2)+'\n')
    print(json.dumps({'configurations':len(specs),'counts':data['counts'],'old':len(old),'new':len(specs)-len(old),
                      'prior_artifacts_verified':len(prior['artifacts'])}))


if __name__=='__main__':main()
