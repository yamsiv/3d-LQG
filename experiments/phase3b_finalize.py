"""Consolidate the completed Phase 3B checkpoint without new simulations.

Prepare exports only saved discovery scores and checks existing archives.
Seal verifies preservation after the permitted report/log edits and creates
a new manifest, leaving the original Phase 3B manifest unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from src.provenance import environment_manifest


ROOT=Path('results/phase3b')
DEST=ROOT/'finalization_20261112'
MUTABLE={'README.md','Agents_Info/RESEARCH_LOG.md',str(ROOT/'STATUS.md'),
         'results/RESEARCH_REPORT_PHASE_3B.md'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(value,indent=2)+'\n')


def prepare():
    DEST.mkdir(exist_ok=True)
    original=json.loads((ROOT/'final_manifest.json').read_text())
    changed=[p for p,r in original['artifacts'].items()
             if not Path(p).exists() or digest(p)!=r['sha256']]
    if changed:raise ValueError(f'Files changed before recovery: {changed}')
    bad_sources=[p for p,h in original['provenance']['source_sha256'].items() if digest(p)!=h]
    if bad_sources:raise ValueError(bad_sources)
    counts={'npz':0,'json':0,'jsonl_rows':0}
    for path in ROOT.rglob('*.npz'):
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:raise ValueError(path)
        counts['npz']+=1
    for path in ROOT.rglob('*.json'):
        json.loads(path.read_text());counts['json']+=1
    for path in ROOT.rglob('*.jsonl'):
        for line in path.read_text().splitlines():
            if line.strip():json.loads(line);counts['jsonl_rows']+=1

    specs=json.loads((ROOT/'extended_design.json').read_text())['configurations']
    coarse=json.loads((ROOT/'screen_analysis/scores.json').read_text())['rows']
    intermediate=json.loads((ROOT/'intermediate_analysis/scores.json').read_text())['rows']
    registry={r['id']:r for r in json.loads((ROOT/'covariance_registry.json').read_text())}
    rows=[]
    for spec in specs:
        identifier=spec['id']
        candidates=[r for r in intermediate if r['id']==identifier and r['eligible_primary']]
        stage='intermediate' if candidates else 'screen'
        if not candidates:candidates=[r for r in coarse if r['id']==identifier and r['eligible_primary']]
        best=min(candidates,key=lambda r:r['robust_score']) if candidates else None
        rows.append(dict(id=identifier,triple=spec['triple'],covariance=spec['covariance'],
            status=registry[identifier]['status'],
            stage=stage if best else 'model_certificate_failure',
            graph_n=8192 if stage=='intermediate' else 2048 if best else None,
            field_n=48 if stage=='intermediate' else 32 if best else None,
            xi=best['xi'] if best else None,gamma=best['gamma'] if best else None,
            robust_J=best['robust_score'] if best else None,
            graph_parents=best['graph_realizations'] if best else 0,
            field_parents=best['field_realizations'] if best else 0))
    save(DEST/'covariance_parameter_map.json',rows)
    with (DEST/'covariance_parameter_map.csv').open('x') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['| ID | Canonical (a,b,c) | Discovery stage | Best tested xi | Linked gamma | Robust J |',
           '|---|---|---|---:|---:|---:|']
    for row in rows:
        triple='('+', '.join(f'{x:.5g}' for x in row['triple'])+')'
        lines.append(f"| {row['id']} | {triple} | {row['stage']} | "+
            (f"{row['xi']:.4g} | {row['gamma']:.5f} | {row['robust_J']:.4f} |" if row['xi'] is not None else '— | — | — |'))
    (DEST/'covariance_parameter_map.md').write_text('\n'.join(lines)+'\n')

    # Each header reports cumulative time since the same path started.
    # The maximum is the final measured checkpoint, not a sum over paired n.
    times={}
    for stage in original['graph_checkpoint_counts']:
        physical={}
        for path in (ROOT/stage).glob('C*/graph_*.npz'):
            with np.load(path) as a:
                h=json.loads(str(a['header']));key=(h['id'],h['seed'])
                physical[key]=max(physical.get(key,0.),h['seconds'])
        times[stage]=dict(physical_paths=len(physical),
                         cumulative_seconds=sum(physical.values()))
    save(DEST/'producer_timings_audited.json',dict(graph_stages=times,
        note='Maximum cumulative header time per path; excludes final checkpoint/path-file writing and model setup. Older summaries selected a size-dependent header and differ slightly; numerical samples and scores are unaffected.'))
    inventory={
        'date':'2026-11-12','recovered_manifest':str(ROOT/'final_manifest.json'),
        'recovered_manifest_sha256':digest(ROOT/'final_manifest.json'),
        'prior_manifest_artifacts_matching_at_recovery':len(original['artifacts']),
        'prior_source_hashes_matching_at_recovery':len(original['provenance']['source_sha256']),
        'integrity_checks':counts,'corrupt_outputs_found':[],
        'completed_validated_components':[
            'Exact biased I3 continuous-minimum generator and graph/metric rules',
            'A2/A3 exact finite-time matrix-grid reference laws (continuous joint minima remain approximate)',
            'Finite-grid unit-log field/GMC/LFPP implementation, 173-test prior suite and saved-data score calculations',
            'All planned calibration, discovery, intermediate, heldout and matched-control batches are complete'],
        'completed_provisional_components':[
            'Finite-resolution gamma(xi) conversion, including invalid-branch/model flags',
            '29 successfully sampled non-reference covariance configurations, with explicit angular/path/minimum error diagnostics',
            'A3/near-A3 candidate region and finite-size metric-measure comparisons'],
        'incomplete_or_unresolved':[
            'C022 produced no graph because its numerical positivity certificate failed; failure is recorded, not an interrupted batch',
            'No controlled arbitrary-covariance continuous excursion/minimum law or asymptotic gamma(xi) map',
            'No identified continuum correspondence curve, scalar level sets or common limit'],
        'optional_omitted_experiments':[
            'Larger covariance sweep or extended optimizer campaign',
            'Higher-precision MCMC/eigenfunction repair or blanket additional realizations',
            'Phase3C larger-graph/field confirmation (recommended, not launched)'],
        'interrupted_primary_batches_found':[],
        'new_simulations_in_this_finalization':0,
        'recovered_subagent_outputs':[
            'docs/PARAMETER_CALIBRATION_PHASE3B.md','docs/PROVISIONAL_SAMPLER_PHASE3B.md',
            'results/phase3b/quality/audited/summary.json',
            'results/phase3b/structure/heldout_independent_review.json'],
    }
    save(DEST/'completion_inventory.json',inventory)
    print(json.dumps({'inventory':str(DEST/'completion_inventory.json'),'checks':counts,
                      'covariance_rows':len(rows),'timings':times}))


def seal():
    output=DEST/'final_manifest.json'
    if output.exists():raise FileExistsError(output)
    prior=json.loads((ROOT/'final_manifest.json').read_text())
    changes=[]
    for path,row in prior['artifacts'].items():
        if digest(path)!=row['sha256']:
            if path not in MUTABLE:raise ValueError(f'Unapproved prior artifact change: {path}')
            changes.append(path)
    for path,sha in prior['provenance']['source_sha256'].items():
        if digest(path)!=sha:raise ValueError(f'Prior source changed: {path}')
    old=json.loads(Path('results/phase3a/final_manifest.json').read_text())
    for path,row in old['artifacts'].items():
        if path not in {'README.md','Agents_Info/RESEARCH_LOG.md','research-notes/questions.md'} and digest(path)!=row['sha256']:
            raise ValueError(f'Phase3A artifact changed: {path}')
    suites=ET.parse(DEST/'tests.xml').findall('.//testsuite')
    tests={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','errors','failures','skipped']}
    if tests['tests']!=173 or tests['errors'] or tests['failures']:raise ValueError(tests)
    docs=[Path(p) for p in MUTABLE]+[DEST/'COMPLETION_INVENTORY.md']
    for path in docs:
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if link.startswith(('http:','https:')):continue
            target=(path.parent/link.split('#')[0]).resolve()
            if not target.exists() and target!=output.resolve():raise ValueError(f'Broken link: {path}: {link}')
    provenance=environment_manifest()
    source=Path(__file__);sha=digest(source);snapshot=DEST/(sha+'.py')
    if not snapshot.exists():snapshot.write_bytes(source.read_bytes())
    artifacts=sorted({p for p in ROOT.rglob('*') if p.is_file() and p!=output}|set(docs))
    save(output,dict(task='Phase3B final consolidation; no new simulations',date='2026-11-12',
        provenance=provenance,tests=tests,
        original_phase3b_manifest=str(ROOT/'final_manifest.json'),
        original_phase3b_manifest_sha256=digest(ROOT/'final_manifest.json'),
        prior_phase3b_artifacts_verified=len(prior['artifacts']),
        permitted_document_revisions=sorted(changes),
        prior_phase3b_source_hashes_unchanged=len(prior['provenance']['source_sha256']),
        raw_data_or_old_source_changes=0,finalization_source_snapshot=str(snapshot),
        artifact_count=len(artifacts),
        artifacts={str(p):{'sha256':digest(p),'bytes':p.stat().st_size} for p in artifacts}))
    print(json.dumps({'output':str(output),'tests':tests,'permitted_document_revisions':changes,
                      'artifacts':len(artifacts),'prior_files_checked':len(prior['artifacts'])}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=['prepare','seal'])
    args=parser.parse_args();prepare() if args.stage=='prepare' else seal()
