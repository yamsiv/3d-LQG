"""Export leading geometry and clearly labeled uncertainty from saved analysis."""
import csv
import json
from pathlib import Path
import numpy as np


def main():
    data=json.loads(Path('results/phase3a/comparison/summary_distributions.json').read_text())
    grid=np.linspace(0,4,1025);rows=[]
    for key,value in data.items():
        c=np.asarray(value['cdf']);balls=np.asarray(value['ball_quantiles'])
        row={'ensemble':key,'realizations':value['realizations'],'physical_median_scale':value['scale']}
        for q in [.1,.25,.75,.9]:row[f'normalized_q{int(100*q)}']=float(np.interp(q,c,grid))
        row.update(tail_mass_above_1_5=float(1-np.interp(1.5,grid,c)),
            ball_mass_r05_q10=float(balls[0,1]),ball_mass_r05_median=float(balls[1,1]),
            ball_mass_r05_q90=float(balls[2,1]),ball_sd_r05=value['ball_sd'][1],
            triangle_small_mean=value['triangle_small_mean'],triangle_middle_mean=value['triangle_middle_mean'],
            zero_distance_atom=value['zero_distance_atom'],scale_cv=value['scale_cv'])
        rows.append(row)
    root=Path('results/phase3a/tables');root.mkdir(exist_ok=True)
    with (root/'leading_geometry.csv').open('x') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    large=json.loads(Path('results/phase3a/large_graph_comparison/summary.json').read_text())
    ci=np.asarray(large['largest_size_paired_flat_minus_selected_mc95'])
    output=root/'largest_graph_intervals_labeled.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps({'source':'results/phase3a/large_graph_comparison/summary.json',
        'pairW1_improvement_mc95':ci[:,0].tolist(),'ballSD_improvement_mc95':ci[:,1].tolist()},indent=2)+'\n')


if __name__=='__main__':main()
