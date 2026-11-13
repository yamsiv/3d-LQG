"""Deterministic finite-rule equivalence and cost check, never holdout data."""
from pathlib import Path
import resource
import time
import numpy as np
from src.lfpp_3d import build_lfpp_graph,root_distances
from src.lfpp_periodic_3c import build_periodic_lfpp_graph
from src.phase3c_fields import write_json_new,sha256_file


def main():
    rows=[]
    for n in[32,48]:
        x,y,z=np.ogrid[:n,:n,:n]
        h=2*np.sin(2*np.pi*x/n)+np.cos(4*np.pi*y/n)-.7*np.sin(2*np.pi*(x+y+z)/n)
        for stencil in['26','primitive2']:
            tick=time.perf_counter();reference,_=build_lfpp_graph(h,.3,stencil=stencil)
            original_time=time.perf_counter()-tick
            tick=time.perf_counter();fast,metadata=build_periodic_lfpp_graph(h,.3,stencil=stencil)
            cold=time.perf_counter()-tick
            assert np.array_equal(fast.indptr,reference.indptr)
            assert np.array_equal(fast.indices,reference.indices)
            maximum=0.
            for start in range(0,len(reference.data),1_000_000):
                old=reference.data[start:start+1_000_000]
                maximum=max(maximum,float(np.max(abs(fast.data[start:start+1_000_000]-old)/old)))
            assert maximum<3e-13
            roots=[0,n+3,n**3-1]
            a=root_distances(reference,roots);tick=time.perf_counter();b=root_distances(fast,roots)
            search=time.perf_counter()-tick
            error=float(abs(a-b).max());assert np.allclose(a,b,rtol=3e-13,atol=2e-14)
            del fast,reference,a,b
            tick=time.perf_counter();warm,_=build_periodic_lfpp_graph(h,.2,stencil=stencil)
            warm_time=time.perf_counter()-tick;del warm
            rows.append({'n':n,'stencil':stencil,'reference_build_seconds':original_time,
                'cold_kernel_build_seconds':cold,'warm_kernel_build_seconds':warm_time,
                'three_root_dijkstra_seconds':search,'maximum_relative_edge_error':maximum,
                'maximum_shortest_path_absolute_error':error,'CSR_bytes':metadata['csr_bytes'],
                'peak_rss_native':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
            print(rows[-1],flush=True)
    directory=Path('results/phase3c/fields/periodic_kernel_validation');directory.mkdir(exist_ok=True)
    write_json_new(directory/'benchmark.json',{'rows':rows,'Gaussian_fields_sampled':0,
        'h_formula':'2sin(2pi*x)+cos(4pi*y)-.7sin(2pi*(x+y+z))',
        'source_sha256':{p:sha256_file(p)for p in['src/lfpp_3d.py','src/lfpp_periodic_3c.py',
             'experiments/phase3c_periodic_benchmark.py','tests/test_lfpp_periodic_3c.py']}})


if __name__=='__main__':main()
