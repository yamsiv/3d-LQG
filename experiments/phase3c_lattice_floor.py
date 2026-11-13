"""Saved-data quantization lower bound for normalized pair-law Wasserstein.

Any graph law is supported on h*integers, h=1/ensemble graph mean. For any
coupling, |X-Y|>=dist(Y,h*integers). Bin midpoint bounds follow from the
1-Lipschitz distance-to-lattice function. This is not a Monte Carlo interval.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from experiments.phase3a_compare import save_json
from experiments.phase3b_search import graph_groups,field_groups


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',default='results/phase3c/tables_final/distance_lattice_floor.json');a=p.parse_args()
    r=Path('results/phase3c');g=graph_groups(sorted((r/'confirmation/graphs').glob('batch_*')),8192)
    f=field_groups([r/'fields/confirmation/log'],48);plan=json.loads((r/'confirmation/frozen_plan.json').read_text());rows=[]
    for i,rs in g.items():
        xi=plan['profiles_frozen_from_discovery'][i]['nominal_xi'];fs=f[(xi,'central')]
        h=1/np.mean([x['root_means'].mean()for x in rs]);sf=np.mean([x['root_means'].mean()for x in fs]);lo=[];hi=[]
        for x in fs:
            xx=x['grid']/sf;c=x['cdfs'].mean(axis=0);w=np.diff(c);mid=(xx[:-1]+xx[1:])/2;half=np.diff(xx)/2;dist=abs(mid-h*np.rint(mid/h))
            lo.append(np.sum(w*np.maximum(0,dist-half)));hi.append(np.sum(w*np.minimum(h/2,dist+half)))
        rows.append(dict(id=i,xi=xi,graph_normalized_hop=h,nearest_lattice_cost_interval=[np.mean(lo),np.mean(hi)],
            interpretation='LowerboundonW1toANYlawongraphintegerhopgrid;intervalfromsavedfieldCDFbins,notMCCI'))
    save_json(a.output,rows)


if __name__=='__main__':main()
