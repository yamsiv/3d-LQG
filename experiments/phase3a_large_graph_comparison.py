"""Reuse archived large exact graphs for pair/ball comparison, no simulation."""
from pathlib import Path
import numpy as np
from experiments.phase3a_compare import field_groups,save_json
from experiments.phase3a_diagnostics import archive_graphs
from src.metric_measure_3a import summarize,discrepancy
from src.provenance import environment_manifest


def main():
    output=Path("results/phase3a/large_graph_comparison");output.mkdir(exist_ok=True)
    if (output/"summary.json").exists():raise FileExistsError(output)
    graphs=archive_graphs()
    fields=field_groups("results/phase3a/confirmation",n=48,stencil="primitive2")
    selected=fields[(48,"primitive2",.3,1.5)];flat=fields[(48,"primitive2",0.,0.)]
    rows=[]
    for n,group in sorted(graphs.items()):
        s=summarize(group)
        rows.append({"n":n,"graphs":len(group),"selected":discrepancy(s,summarize(selected)),
            "flat":discrepancy(s,summarize(flat))})
    rng=np.random.default_rng(202611113933);target=graphs[max(graphs)];draws=[]
    for _ in range(160):
        gi=rng.integers(len(target),size=len(target));fi=rng.integers(len(flat),size=len(flat))
        g=summarize([target[i]for i in gi])
        a=discrepancy(g,summarize([flat[i]for i in fi]));b=discrepancy(g,summarize([selected[i]for i in fi]))
        draws.append([a['pair_w1']-b['pair_w1'],a['ball_dispersion_l1']-b['ball_dispersion_l1']])
    np.savez_compressed(output/'paired_bootstrap.npz',draws=np.array(draws))
    save_json(output/'summary.json',{'provenance':environment_manifest(),'rows':rows,
        'largest_size_paired_flat_minus_selected_mc95':np.quantile(draws,[.025,.975],axis=0),
        'columns':['pairW1','ballSD'],
        'qualification':'Uses648 archived independent graphs; onlypair/ballinformationexistsatlargesize, not8pointmatrices. Root/graphfinite-sizeeffectsremain.'})
    print('Saved large-graph comparison without generating graphs.')


if __name__=='__main__':main()
