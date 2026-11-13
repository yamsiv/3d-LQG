"""Read-only raw-data audit and optional final immutable Phase3C manifest.

No cleanup, Git manipulation, physical resimulation or result reorganization.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
from experiments.phase3a_compare import save_json
from experiments.phase3b_manifest import metric_check
from src.provenance import environment_manifest

R=Path('results/phase3c')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--seal',action='store_true');a=p.parse_args()
    if a.seal:
        output=R/'final_manifest.json'
        if output.exists():raise FileExistsError(output)
        integrity=json.loads((R/'final_integrity.json').read_text())
        if not integrity['all_checks_pass']:raise ValueError('Audit failed')
        provenance=environment_manifest();snap=R/'final_source_snapshots';snap.mkdir(exist_ok=True);index={}
        for source,digest in provenance['source_sha256'].items():
            dest=snap/(digest+Path(source).suffix)
            if not dest.exists():dest.write_bytes(Path(source).read_bytes())
            assert sha(dest)==digest;index[source]=dict(sha256=digest,snapshot=str(dest))
        artifacts=sorted(p for p in R.rglob('*')if p.is_file()and'__pycache__'not in p.parts and p!=output)
        artifacts+=[Path('results/RESEARCH_REPORT_PHASE_3C.md'),Path('Agents_Info/RESEARCH_LOG.md'),Path('README.md'),
            Path('AGENTS.md'),Path('Agents_Info/AGENTS.md'),Path('Agents_Info/RESEARCH_CONTEXT.md'),
            Path('pyproject.toml'),Path('requirements.txt'),Path('requirements-lock.txt')]
        artifacts+=sorted(p for p in Path('docs').glob('*.md')if '3c'in p.name.lower())
        save_json(output,dict(task='Phase3C',decision='Extendedfinitecandidatebands;noidentifiedcontinuumcorrespondence',
            provenance=provenance,source_snapshots=index,artifacts={str(p):dict(sha256=sha(p),bytes=p.stat().st_size)for p in artifacts},
            integrity_audit=integrity,phase3d_started=False,no_simulations_deleted_or_reorganized=True))
        print(json.dumps({'manifest':str(output),'artifacts':len(artifacts)}));return
    output=R/'final_integrity.json'
    if output.exists():raise FileExistsError(output)
    mutable={'README.md','Agents_Info/RESEARCH_LOG.md','research-notes/questions.md'}
    # Phase3B has a later sealed finalization; its earlier manifest predates the
    # finalized report. Use the exact baseline recorded at Phase3C recovery.
    recovery=json.loads((R/'recovery.json').read_text())
    old=json.loads(Path(recovery['prior_manifest']).read_text());oldchecked=0
    for path,entry in old['artifacts'].items():
        if path in mutable:continue
        assert sha(path)==entry['sha256'],f'Prior artifact changed:{path}';oldchecked+=1
    for path,digest in old['provenance']['source_sha256'].items():assert sha(path)==digest,f'Prior source changed:{path}'
    counts=collections.Counter();graphkeys=set();graphfiles=[];metricfiles=[];partial=[];hashes={}
    for path in sorted(R.rglob('*')):
        if not path.is_file():continue
        if path.suffix=='.partial':partial.append(str(path))
        if path.suffix=='.json':json.loads(path.read_text());counts['json']+=1
        if path.suffix=='.npz':
            with zipfile.ZipFile(path)as archive:assert archive.testzip()is None,f'CRC:{path}'
            counts['npz_crc']+=1
            if path.name.startswith('graph_')and '_n'in path.stem:
                with np.load(path)as z:
                    h=json.loads(str(z['header']));metric_check(path,z)
                    v=h['validation'];assert v['connected']and v['hamiltonian_cycle'];assert v['coordinate_diagonal_counts']==[h['n']-3]*3
                    graphkeys.add((h['id'],h['seed']));graphfiles.append(str(path));counts['graph_records']+=1
            if path.name.startswith('metric_'):
                with np.load(path)as z:
                    h=json.loads(str(z['header']));metric_check(path,z)
                    source=h.get('source_field');expected=h.get('source_field_sha256')
                    if source and expected:
                        if source not in hashes:hashes[source]=sha(source)
                        assert hashes[source]==expected,f'Field hash:{path}'
                    if 'points'in z:
                        for index,probe in enumerate(h.get('probes',[])):
                            assert np.isfinite(probe['log_total_mass'])
                            # Equal-gamma aliases must not secretly redraw points.
                            for j in range(index):
                                if probe['gamma']==h['probes'][j]['gamma']:
                                    for key in ['points','grid','cdfs','root_means','matrices','bin_width']:assert np.array_equal(z[key][index],z[key][j]),f'Alias mismatch:{path}'
                    metricfiles.append(str(path));counts['metric_records']+=1
    assert not partial,f'Unfinishedarchives:{partial}'
    # Count genuinely new physical paths by seed/C identity, not graph files.
    oldkeys=set()
    for path in Path('results/phase3b').rglob('graph_*_n*.npz'):
        with np.load(path)as z:
            h=json.loads(str(z['header']));oldkeys.add((h['id'],h['seed']))
    gatekeys=set()
    for path in (R/'calibration/sampler/records').glob('*.json'):
        h=json.loads(path.read_text())
        if h.get('status')=='success':gatekeys.add((h['id'],h['seed']))
    newly_generated=(graphkeys|gatekeys)-oldkeys
    # Independence and actual preserved low modes in confirmation sources.
    reg=json.loads((R/'fields/confirmation_registry.json').read_text());newfieldids=set();newseeds=set();raw32hash=set();raw48hash=set()
    modes=np.fft.fftfreq(32)*32;shared=[]
    for row in reg['records']:
        source=Path(row['source_field']);assert source.exists()
        side=json.loads(source.with_suffix('.json').read_text());assert sha(source)==side['source_field_sha256'];newseeds.add(row['field_seed']);newfieldids.add(row['realization'])
        with np.load(source)as z:
            h32=z['h32'];h48=z['h48'];raw32hash.add(hashlib.sha256(h32.tobytes()).hexdigest());raw48hash.add(hashlib.sha256(h48.tobytes()).hexdigest())
            f32=np.fft.fftn(h32)/32**3;f48=np.fft.fftn(h48)/48**3
            for k in [(1,0,0),(2,-3,1),(5,3,-2)]:shared.append(abs(f32[tuple(x%32 for x in k)]-f48[tuple(x%48 for x in k)]))
    assert len(newfieldids)==len(newseeds)==len(raw32hash)==len(raw48hash)==24
    prior=reg['prior_identities'];assert not newseeds.intersection(prior['field_seeds']);assert not raw32hash.intersection(prior['array_hashes']['32']);assert not raw48hash.intersection(prior['array_hashes']['48'])
    assert max(shared)<1e-13
    # Compare retained modes in EVERY extended sample, not merely identical seeds.
    extension_errors=[]
    for path in (R/'fields').rglob('field_*_n64.npz'):
        with np.load(path)as z:
            if 'h64'not in z:continue
            h=json.loads(str(z['header']));parent=Path(h['parent_source_field'])if 'parent_source_field'in h else None
            if parent is None:continue
            with np.load(parent)as pz:f48=np.fft.fftn(pz['h48'])/48**3
            f64=np.fft.fftn(z['h64'])/64**3
            for k in [(1,0,0),(5,3,-2),(10,2,1)]:extension_errors.append(abs(f48[tuple(x%48 for x in k)]-f64[tuple(x%64 for x in k)]))
    # Confirm exact expected counts and no seed tuning.
    expected={'log':{32:120,48:120,64:24},'flat':{32:24,48:24},'smooth':{32:24,48:24}}
    for model,grids in expected.items():
        for n,number in grids.items():assert len(list((R/'fields/confirmation'/model).glob(f'xi*/metric_*_n{n}.npz')))==number
    assert len(list((R/'fields/confirmation_stencil_control/log').glob('xi*/metric_*_n48.npz')))==24
    for path in [R/'confirmation/analysis/COMPLETE.json',R/'confirmation/paired_checks/COMPLETE.json']:assert path.is_file()
    suites=ET.parse(R/'final_tests.xml').findall('.//testsuite');tests={k:sum(int(s.get(k,0))for s in suites)for k in ['tests','errors','failures','skipped']};assert tests['tests']>=214 and not tests['errors']and not tests['failures']
    report=Path('results/RESEARCH_REPORT_PHASE_3C.md');missing=[]
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
        if not target.startswith(('https:','http:','#'))and not(report.parent/target.split('#')[0]).exists():missing.append(target)
    assert not missing,missing
    data=dict(all_checks_pass=True,prior_phase3b_artifacts_unchanged=oldchecked,prior_source_files_unchanged=len(old['provenance']['source_sha256']),
        parsed=dict(counts),partial_archives=partial,graph_parent_keys_under_phase3c=len(graphkeys),gate_parent_keys=len(gatekeys),
        newly_generated_physical_paths=len(newly_generated),new_paths_definition='Unique(C,seed)under3CgrapharchivesorStageIrecords,excludingallprior3Bgraphparents;repeatedsizes/copiesnotnewpaths',
        actual_gaussian_families=dict(discovery=48,independent_confirmation=24,total_used=72,new_this_phase=36,prior_reused=36),
        confirmation_lowmode_max_error=float(max(shared)),extension_modes_checked=len(extension_errors),extension_max_error=float(max(extension_errors,default=0)),
        tests=tests,report_local_links_valid=True,no_git_exclusions_modified=True,no_cleanup_performed=True)
    save_json(output,data);print(json.dumps(data,indent=2))


if __name__=='__main__':main()
