import argparse
import json
from pathlib import Path
import pytest
import experiments.phase3c_confirmation_fields as producer


def test_unfrozen_plan_never_authorizes_holdout(tmp_path):
    path=tmp_path/'plan.json'
    path.write_text(json.dumps({'stage':'independent_confirmation','status':'awaiting_freeze'}))
    with pytest.raises(ValueError,match='Freeze'):
        producer.frozen_config(path)


@pytest.mark.parametrize('enable_n64',[False,True])
def test_controller_reaches_sampling_guard_without_sampling(tmp_path,monkeypatch,enable_n64):
    # Stop at a mocked materialization boundary: noGaussian,metricorN64 sample.
    pilot=tmp_path/'pilot.json';pilot.write_text(json.dumps({'peak_rss_native':100,'seconds':1.}))
    config={'stage':'independent_confirmation','status':'frozen_by_coordinator',
        'metric_jobs_authorized':True,'hypotheses':[{'test_fixture':True}],
        'xis':[.2,.225,.25],'gamma_rule':'modelwise_lambda','smooth_xis':[],
        'smooth_field_ids':[],'enable_n64':enable_n64,
        'n64_xis':[.2,.225,.25]if enable_n64 else None,
        'n64_field_ids':[1000],'n64_deterministic_pilot':str(pilot)}
    path=tmp_path/'plan.json';path.write_text(json.dumps(config))
    registry={'records':[{'realization':1000}], 'prior_identities':{'source_sha256':{}}}
    monkeypatch.setattr(producer,'plan',lambda *args:registry)
    monkeypatch.setattr(producer,'source_hashes',lambda:{})
    monkeypatch.setattr(producer,'environment_manifest',lambda:{})
    monkeypatch.setattr(producer,'sha256_file',lambda path:'unit_test_hash')
    def no_sampling(*args):raise RuntimeError('STOP_before_sampling')
    monkeypatch.setattr(producer,'materialize',no_sampling)
    args=argparse.Namespace(config=str(path),field_ids=[1000],sizes=[64]if enable_n64 else[32],
        models=['log'],xis=None,registry='unused',seed=202611123900,
        output=str(tmp_path/'output'),label='unit_guard')
    with pytest.raises(RuntimeError,match='STOP_before_sampling'):
        producer.run(args)
