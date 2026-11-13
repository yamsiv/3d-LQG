"""Analytic deterministic N64 resource check;never sample a Gaussian field."""
from pathlib import Path
import resource
import time
import numpy as np
from src.field_metrics_3c import metric_arrays
from src.log_field_3d import spectral_configuration
from src.phase3c_fields import write_json_new,sha256_file


def main():
    n=64;x,y,z=np.ogrid[:n,:n,:n]
    h=2*np.sin(2*np.pi*x/n)+np.cos(4*np.pi*y/n)-.7*np.sin(2*np.pi*(x+y+z)/n)
    variance=spectral_configuration(n)[2]
    tick=time.perf_counter()
    arrays,_,metadata,counts=metric_arrays(h,.225,variance,
        [('central',.9),('low',.8),('high',1.)],np.linspace(.015,.985,16),stencil='primitive2')
    elapsed=time.perf_counter()-tick
    assert arrays['matrices'].shape==(3,16,16)
    assert np.allclose(arrays['matrices'],arrays['matrices'].transpose(0,2,1))
    result={'n':64,'deterministic_analytic_grid':True,'Gaussian_fields_generated':0,
        'holdout_fields_sampled':0,'seconds':elapsed,
        'peak_rss_native':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'stencil':'primitive2','unique_gamma_rows':counts['new_unique_gamma_rows'],
        'CSR_bytes':metadata['csr_bytes'],
        'source_sha256':{p:sha256_file(p)for p in['experiments/phase3c_n64_preflight.py',
            'src/field_metrics_3c.py','src/lfpp_periodic_3c.py']}}
    directory=Path('results/phase3c/fields/confirmation_preparation');directory.mkdir(exist_ok=True)
    write_json_new(directory/'n64_deterministic_pilot.json',result)
    print(result)


if __name__=='__main__':main()
