"""Predeclared finite-feature family analysis for Phase 3C.

Observed low-discrepancy components are a finite sampling graph, not a theorem
about continuous covariance curves. Whole physical realizations remain the
statistical units. Sampler-systematic budgets are separate from MC tolerances.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from src.metric_features_3b import (
    feature_vector, grouped_feature_distance, DEFAULT_WEIGHTS, _gauge_records,
)


def sampled_matrix_panels(records, *, gauge="grounded", normalization="mean", points=16):
    """One labeled upper-triangle panel per parent, in common ensemble units.

    Iid sampled point labels make the population panel law exchangeable. The
    finite estimate need not be invariant under separately relabeling each panel;
    a common point permutation is an exact coordinate permutation. Repeated
    sampled locations and their off-diagonal zero distances are retained.
    """
    records=_gauge_records(list(records),gauge,False)
    if not records:raise ValueError("No physical parent panels")
    if normalization not in("mean","median"):raise ValueError("Unknown normalization")
    if normalization=="mean":
        scale=float(np.mean([np.mean(r["root_means"])for r in records]))
    else:
        from src.metric_measure_3a import ensemble_median
        scale=ensemble_median(records)
    if scale<=0 or not np.isfinite(scale):raise ValueError("Positive finite ensemble scale required")
    columns=np.triu_indices(points,1)
    panels=[]
    for record in records:
        matrix=np.asarray(record["matrix"],float)
        if (matrix.shape!=(points,points)or not np.all(np.isfinite(matrix))or np.any(matrix<0)
            or not np.allclose(matrix,matrix.T)or not np.allclose(np.diag(matrix),0)):
            raise ValueError("Matched finite symmetric sampled distance matrices required")
        panels.append(matrix[columns]/scale)
    return {"vectors":np.array(panels),"scale":scale,"points":points,"coordinates":len(columns[0]),
            "gauge":gauge,"normalization":normalization}


def signed_matrix_energy(left_vectors,right_vectors):
    """Signed energy U-form in RMS Euclidean distance on labeled panels.

    E_U=2*crossmean-within_left_offdiagonal_mean-within_right_offdiagonal_mean.
    Negative finite estimates are retained. Estimated common ensemble scales
    prevent calling this an exactly unbiased population-kernel statistic.
    """
    from scipy.spatial.distance import cdist
    left,right=np.asarray(left_vectors,float),np.asarray(right_vectors,float)
    if (left.ndim!=2 or right.ndim!=2 or left.shape[1]!=right.shape[1]
        or min(len(left),len(right))<2 or left.shape[1]<1
        or not np.all(np.isfinite(left))or not np.all(np.isfinite(right))):
        raise ValueError("Need>=2 finite parent vectors on each side with matched coordinates")
    scale=np.sqrt(left.shape[1]);m,n=len(left),len(right)
    cross=float(cdist(left,right).mean()/scale)
    a=cdist(left,left);b=cdist(right,right)
    within_left=float((a.sum()-np.trace(a))/(m*(m-1))/scale)
    within_right=float((b.sum()-np.trace(b))/(n*(n-1))/scale)
    return {"energy_u":2*cross-within_left-within_right,"cross_mean":cross,
            "within_left_u_mean":within_left,"within_right_u_mean":within_right,
            "parent_counts":[m,n],"panel_coordinates":left.shape[1],
            "qualification":"signed finite U-form; no clipping/square root; common estimated scales make it descriptive, not an exactly unbiased kernel test"}


def matrix_energy_comparison(left_records,right_records,rng=None,*,draws=32,
                             gauge="grounded",normalization="mean",points=16,
                             left_indices=None,right_indices=None):
    """Full panel-law diagnostic, with optional whole-parent bootstrap.

    Supply shared right_indices for log/smooth/flat comparisons on common fields;
    share left_indices for the same graph target. Gauge and ensemble scale are
    recomputed inside every draw. Neither epsilon nor the primary J is changed.
    """
    left_records,right_records=list(left_records),list(right_records)
    kwargs=dict(gauge=gauge,normalization=normalization,points=points)
    left=sampled_matrix_panels(left_records,**kwargs);right=sampled_matrix_panels(right_records,**kwargs)
    result=signed_matrix_energy(left["vectors"],right["vectors"])
    result.update(left_scale=left["scale"],right_scale=right["scale"],gauge=gauge,normalization=normalization)
    if draws:
        if left_indices is None or right_indices is None:
            if rng is None:raise ValueError("Supply rng or both parent resampling arrays")
            if left_indices is None:left_indices=rng.integers(len(left_records),size=(draws,len(left_records)))
            if right_indices is None:right_indices=rng.integers(len(right_records),size=(draws,len(right_records)))
        left_indices,right_indices=np.asarray(left_indices,int),np.asarray(right_indices,int)
        if left_indices.shape!=(draws,len(left_records))or right_indices.shape!=(draws,len(right_records)):
            raise ValueError("Resampling arrays must have shape draws x physical parents")
        if (np.any(left_indices<0)or np.any(left_indices>=len(left_records))
            or np.any(right_indices<0)or np.any(right_indices>=len(right_records))):
            raise ValueError("Parent resampling index out of range")
        energies=[]
        for a,b in zip(left_indices,right_indices):
            l=sampled_matrix_panels([left_records[i]for i in a],**kwargs)
            r=sampled_matrix_panels([right_records[i]for i in b],**kwargs)
            energies.append(signed_matrix_energy(l["vectors"],r["vectors"])["energy_u"])
        result.update(mc95=np.quantile(energies,[.025,.975]).tolist(),bootstrap_energies=energies,
                      bootstrap_draws=draws,
                      independence_unit="whole physical parent panel; normalizer and gauge recomputed")
    return result


WEIGHTINGS = {
    "primary": dict(DEFAULT_WEIGHTS),
    "equal_families": {family: 1. for family in DEFAULT_WEIGHTS},
    "higher_point": dict(DEFAULT_WEIGHTS, pair=.5, balls=1., triangles=1.5,
                         quartets=2., matrix_dependence=1.5),
    "without_mesoscopic": dict(DEFAULT_WEIGHTS, mesoscopic=0., atoms=0.),
}
PRIMARY_NORMALIZATION = "mean"
PRIMARY_GAUGE = "grounded"
ADJACENCY = {"covariance_radius": .25, "covariance_k": 4,
             "maximum_xi_gap": .0375, "diagonal_edges": False,
             "radius_sensitivity": [.18, .25, .32]}


def ensemble_features(records, *, gauge="grounded", normalization="mean"):
    """Reuse the47 declared features with mean-scale primary normalization.

    The legacy label pair_mean_over_median stores mean/selected_scale. It is
    identically1 with mean normalization, and is retained as a normalization
    check for exact compatibility with the old feature layout. Other features
    and group weights are unchanged. Median normalization is a sensitivity.
    """
    if normalization not in ("mean", "median"):
        raise ValueError("normalization must be mean or median")
    records = _gauge_records(list(records), gauge, False)
    if not records:
        raise ValueError("No physical realizations")
    scale = None
    if normalization == "mean":
        scale = float(np.mean([np.mean(record["root_means"]) for record in records]))
    features = feature_vector(records, gauge=gauge, already_transformed=True, scale=scale)
    features["diagnostics"]["normalization_rule"] = normalization
    features["diagnostics"]["legacy_pair_mean_label_is_mean_over_selected_scale"] = True
    return features


def _valid_delta(left, right):
    if left["names"] != right["names"] or left["families"] != right["families"]:
        raise ValueError("Feature layouts differ")
    valid = np.asarray(left["valid"], bool) & np.asarray(right["valid"], bool)
    return np.asarray(left["vector"])-np.asarray(right["vector"]), valid


def _weighted_group_score(components, weights):
    total_weight = sum(weights.get(key, 0.) for key in components)
    if total_weight <= 0:
        raise ValueError("No positively weighted valid feature families")
    return float(sum(weights.get(key, 0.)*value for key,value in components.items())/total_weight)


def calibrate_split_tolerance(cohorts, rng, *, draws=32, quantile=.95):
    """Fix tolerances from disjoint same-model splits of OLD physical records.

    Every draw partitions each cohort without replacement into balanced halves.
    Repeated partitions reuse a dataset and are dependent. Epsilon is the maximum
    cohort-specific95th percentile, computed separately per gauge, normalization
    and weighting. It is a conservative smaller-parent-count descriptive noise
    reference, not an acceptance test, bias bound or asymptotic tolerance.
    There is deliberately no unproved sample-count rescaling.
    """
    if draws < 2 or not 0 < quantile < 1:
        raise ValueError("Need>=2 partitions and a proper quantile")
    if not cohorts or any(len(rows) < 4 for rows in cohorts.values()):
        raise ValueError("Each cohort requires>=4 independent physical records")
    results = {}
    # The same partitions in both gauges/normalizations preserve their coupling.
    partitions = {};shared={};partition_cache={}
    for name,rows in cohorts.items():
        identities=[]
        for index,row in enumerate(rows):
            header=row.get("header",{})
            identity=header.get("source_field_sha256",header.get("physical_path_id"))
            if identity is None and header.get("field_seed")is not None:
                identity=("field",header["field_seed"])
            if identity is None and header.get("seed")is not None:
                identity=(header.get("id",name),header["seed"])
            if identity is None:identity=(name,index)
            identities.append(str(identity))
        key=tuple(identities)
        if key not in partition_cache:
            partition_cache[key]=[rng.permutation(len(rows)) for _ in range(draws)]
        partitions[name]=partition_cache[key]
        shared.setdefault(key,[]).append(name)
    for normalization in ("mean", "median"):
        for gauge in ("grounded", "unit_mass"):
            by_cohort = {}
            layout = None
            for name,records in cohorts.items():
                values = {weight:[] for weight in WEIGHTINGS}
                signed, masks, family_scores = [], [], []
                for permutation in partitions[name]:
                    cut = len(records)//2
                    left = ensemble_features([records[i] for i in permutation[:cut]],
                                             gauge=gauge, normalization=normalization)
                    right = ensemble_features([records[i] for i in permutation[cut:]],
                                              gauge=gauge, normalization=normalization)
                    layout = left
                    score = grouped_feature_distance(left, right)
                    for weight,settings in WEIGHTINGS.items():
                        values[weight].append(_weighted_group_score(score["components"],settings))
                    delta, valid = _valid_delta(left,right)
                    signed.append(delta);masks.append(valid);family_scores.append(score["components"])
                signed, masks = np.asarray(signed),np.asarray(masks)
                feature95=[]
                for index in range(signed.shape[1]):
                    eligible = abs(signed[masks[:,index],index])
                    feature95.append(float(np.quantile(eligible,quantile)) if len(eligible) else 0.)
                families = {}
                for family in dict.fromkeys(layout["families"]):
                    samples = [row[family] for row in family_scores if family in row]
                    families[family] = float(np.quantile(samples,quantile)) if samples else None
                by_cohort[name]={"parent_count":len(records),
                                 "half_counts":[len(records)//2,len(records)-len(records)//2],
                                 "scores":values,
                                 "epsilon_by_weight":{key:float(np.quantile(v,quantile))for key,v in values.items()},
                                 "signed_feature_differences":signed,"valid":masks,
                                 "feature_abs95":feature95,"family95":families}
            feature_floor=.05*np.asarray(layout["scales"])
            epsilon={weight:max(row["epsilon_by_weight"][weight]for row in by_cohort.values())
                     for weight in WEIGHTINGS}
            family_threshold={}
            for family in dict.fromkeys(layout["families"]):
                valid_values=[row["family95"][family]for row in by_cohort.values()
                              if row["family95"][family] is not None]
                family_threshold[family]=max(valid_values)if valid_values else None
            feature_threshold=np.maximum(feature_floor,np.max(
                [row["feature_abs95"]for row in by_cohort.values()],axis=0))
            results[f"{gauge}:{normalization}"]={"epsilon":epsilon,"cohorts":by_cohort,
                "names":layout["names"],"families":layout["families"],"natural_scales":layout["scales"],
                "feature_abs_threshold":feature_threshold,"feature_floor":feature_floor,
                "family_threshold":family_threshold}
    return {"formula":f"epsilon=max_cohort quantile_{quantile:g}(J of{draws} balanced disjoint within-cohort splits)",
            "draws":draws,"quantile":quantile,"sample_count_rescaling":False,
            "shared_partition_cohorts":list(shared.values()),
            "results":results,
            "qualification":"conditional dependent partitions of old data; smaller-parent-count descriptive tolerance; MC does not bound numerical sampler bias"}


def compare_features(left, right, calibration, *, gauge="grounded", normalization="mean",
                     weighting="primary", sampler_allowed=None, calibration_allowed=True):
    """Signed valid residuals and component flags at the frozen tolerance.

    A low aggregate J is not sufficient: component/feature flags expose tradeoffs.
    Unknown/failed sampler gates never become trusted candidate nodes, even with
    a low point score. Threshold flags are unadjusted diagnostics, not formal
    multiple-testing decisions. Keep systematic budgets separate.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError("Unknown predeclared weighting")
    reference=calibration["results"][f"{gauge}:{normalization}"]
    if left["names"]!=reference["names"]:
        raise ValueError("Calibration feature layout differs")
    discrepancy=grouped_feature_distance(left,right,weights=WEIGHTINGS[weighting])
    delta,valid=_valid_delta(left,right)
    epsilon=float(reference["epsilon"][weighting])
    threshold=np.asarray(reference["feature_abs_threshold"])
    flags=(abs(delta)>threshold)&valid
    family_flags={family:(reference["family_threshold"][family] is not None and
                          value>reference["family_threshold"][family])
                  for family,value in discrepancy["components"].items()}
    low=discrepancy["total"]<=epsilon
    eligible=bool(isinstance(sampler_allowed,(bool,np.bool_))and sampler_allowed and calibration_allowed)
    return {"score":discrepancy["total"],"epsilon":epsilon,"point_low_without_gate":low,
            "eligible":eligible,"candidate":bool(low and eligible),
            "sampler_allowed":sampler_allowed,"calibration_allowed":bool(calibration_allowed),
            "signed_residuals":delta,"valid":valid,"feature_flags":flags,
            "family_components":discrepancy["components"],"family_flags":family_flags,
            "componentwise_low":bool(not any(family_flags.values())),
            "gauge":gauge,"normalization":normalization,"weighting":weighting,
            "threshold_status":"predeclared finite-noise diagnostic; not a continuum acceptance criterion"}


def bootstrap_ensemble_features(records, rng=None, *, indices=None, draws=32,
                                gauge="grounded", normalization="mean"):
    """Whole-parent feature draws; supply common indices for coupled parameters.

    Sorting/physical identity matching across parameter records is the loader's
    responsibility. No roots or matrix subsets are resampled as independent
    parents. The ensemble mean/median is recalculated inside every draw.
    """
    records=list(records)
    if not records:raise ValueError("No physical parents")
    if indices is None:
        if rng is None:raise ValueError("Supply rng or explicit common parent indices")
        indices=rng.integers(len(records),size=(draws,len(records)))
    indices=np.asarray(indices,int)
    if indices.ndim!=2 or indices.shape[1]!=len(records)or np.any(indices<0)or np.any(indices>=len(records)):
        raise ValueError("Expected draws x parents resampling indices")
    point=ensemble_features(records,gauge=gauge,normalization=normalization)
    samples=[ensemble_features([records[i]for i in row],gauge=gauge,normalization=normalization)
             for row in indices]
    return {"point":point,"vectors":np.array([r["vector"]for r in samples]),
            "valid":np.array([r["valid"]for r in samples]),"indices":indices,
            "independence_unit":"whole physical graph/field; parameter choices share supplied indices"}


def observed_family_components(nodes, *, epsilon, covariance_radius=.25, covariance_k=4,
                               maximum_xi_gap=.0375, diagonal_edges=False,
                               score_key="score", minimum_covariances=4,
                               minimum_xi_values=3, minimum_xi_extent=.05,
                               tested_covariances=None):
    """Connected low-J components on the OBSERVED product-neighbor graph.

    Primary edges move either between mutual-kNN covariance neighbors at the
    same measured xi, or between adjacent measured xi at a fixed covariance.
    High-J/failed intermediate nodes are retained when defining adjacency, then
    removed; they cannot be jumped over. Diagonal edges are sensitivity only.
    Components describe finite neighbor support, never unobserved continuity.
    """
    nodes=list(nodes)
    if not nodes:return {"components":[],"edges":[],"low_node_indices":[]}
    if covariance_radius<=0 or covariance_k<1 or maximum_xi_gap<=0:
        raise ValueError("Positive neighborhood parameters required")
    triples={}
    by_covariance=defaultdict(list)
    for spec in tested_covariances or []:
        triples[str(spec["id"])]=np.asarray(spec["triple"],float)
    for index,node in enumerate(nodes):
        identifier=str(node["id"])
        triple=np.asarray(node["triple"],float)
        if triple.shape!=(3,) or not np.all(np.isfinite(triple)):
            raise ValueError("Every node requires three finite covariance entries")
        if identifier in triples and not np.allclose(triples[identifier],triple,atol=1e-12,rtol=0):
            raise ValueError("One covariance ID has inconsistent triples")
        triples[identifier]=triple;by_covariance[identifier].append(index)
    identifiers=sorted(triples)
    points=np.array([triples[i]for i in identifiers])
    distances=np.linalg.norm(points[:,None,:]-points[None,:,:],axis=2)
    np.fill_diagonal(distances,np.inf)
    nearest={i:set(np.argsort(distances[i])[:min(covariance_k,len(identifiers)-1)])
             for i in range(len(identifiers))}
    neighbors={(identifiers[i],identifiers[j])for i,j in combinations(range(len(identifiers)),2)
               if distances[i,j]<=covariance_radius and j in nearest[i] and i in nearest[j]}
    adjacent_xi=set()
    for indices in by_covariance.values():
        order=sorted(indices,key=lambda i:nodes[i]["xi"])
        if len({float(nodes[i]["xi"])for i in order})!=len(order):
            raise ValueError("One covariance has duplicate xi observations")
        for a,b in zip(order[:-1],order[1:]):
            if float(nodes[b]["xi"])-float(nodes[a]["xi"])<=maximum_xi_gap+1e-12:
                adjacent_xi.add((min(a,b),max(a,b)))
    low={i for i,node in enumerate(nodes)if node.get("eligible",False)
         and np.isfinite(node[score_key])and node[score_key]<=epsilon}
    edges=[];adjacency={i:set()for i in low}
    for a,b in combinations(sorted(low),2):
        left,right=nodes[a],nodes[b]
        if left["id"]==right["id"]:
            linked=(a,b)in adjacent_xi
        else:
            pair=tuple(sorted((str(left["id"]),str(right["id"]))))
            gap=abs(float(left["xi"])-float(right["xi"]))
            linked=pair in neighbors and (gap<=1e-12 or(diagonal_edges and gap<=maximum_xi_gap+1e-12))
        if linked:
            adjacency[a].add(b);adjacency[b].add(a);edges.append([a,b])
    unseen=set(low);components=[]
    while unseen:
        start=min(unseen);stack=[start];component=[];unseen.remove(start)
        while stack:
            current=stack.pop();component.append(current)
            for neighbor in adjacency[current]&unseen:
                unseen.remove(neighbor);stack.append(neighbor)
        component.sort();covariances=sorted({str(nodes[i]["id"])for i in component})
        xis=sorted({float(nodes[i]["xi"])for i in component})
        gamma=[float(nodes[i]["gamma"])for i in component if nodes[i].get("gamma")is not None]
        extent=xis[-1]-xis[0]
        family=len(covariances)>=minimum_covariances and len(xis)>=minimum_xi_values and extent>=minimum_xi_extent-1e-12
        components.append({"indices":component,"covariance_ids":covariances,
                           "xi_values":xis,"xi_extent":extent,
                           "gamma_extent":[min(gamma),max(gamma)]if gamma else None,
                           "observed_family_support":bool(family),
                           "minimum_score":min(float(nodes[i][score_key])for i in component),
                           "warning":"finite observed adjacency only; missing regions are not interpolated"})
    components.sort(key=lambda c:(-len(c["covariance_ids"]),-c["xi_extent"],c["minimum_score"]))
    return {"components":components,"edges":edges,"low_node_indices":sorted(low),
            "neighbor_definition":{"covariance_radius":covariance_radius,"covariance_k":covariance_k,
                                    "maximum_xi_gap":maximum_xi_gap,"diagonal_edges":diagonal_edges},
            "family_support_definition":{"minimum_covariances":minimum_covariances,
                                         "minimum_xi_values":minimum_xi_values,"minimum_xi_extent":minimum_xi_extent},
            "epsilon":epsilon,"requires_explicit_eligible_mask":True}


def component_residual_profiles(nodes, components, names, scales):
    """Equal-covariance signed profiles; repeated xi are not independent data.

    Persistent same-sign flags count AGAINST agreement. They are not rewarded
    as a match merely because residuals align along a region. Fractions are
    descriptive, with equal covariance weights and no binomial interpretation.
    """
    scale=np.asarray(scales,float)
    result=[]
    for component in components:
        by_covariance=defaultdict(list)
        for index in component["indices"]:
            node=nodes[index]
            if node.get("signed_residuals") is not None:
                by_covariance[str(node["id"])].append(node)
        summaries=[]
        for identifier,rows in by_covariance.items():
            delta=np.asarray([r["signed_residuals"]for r in rows],float)
            valid=np.asarray([r["valid"]for r in rows],bool)
            flags=np.asarray([r.get("feature_flags",np.zeros(len(scale),bool))for r in rows],bool)&valid
            count=valid.sum(axis=0)
            mean=np.sum(np.where(valid,delta,0.),axis=0)/np.maximum(count,1)
            flag_fraction=flags.sum(axis=0)/np.maximum(count,1)
            positive=np.sum(valid&(delta>0),axis=0)/np.maximum(count,1)
            summaries.append((mean,flag_fraction,positive,count>0))
        if not summaries:
            result.append(dict(component_indices=component["indices"],status="no_signed_records"));continue
        mean=np.array([r[0]for r in summaries]);flag=np.array([r[1]for r in summaries])
        pos=np.array([r[2]for r in summaries]);valid=np.array([r[3]for r in summaries])
        count=valid.sum(axis=0)
        average=lambda a:np.sum(np.where(valid,a,0.),axis=0)/np.maximum(count,1)
        flag_fraction=average(flag);positive_fraction=average(pos)
        persistent=(flag_fraction>=.75)&((positive_fraction>=.75)|(positive_fraction<=.25))&(count>0)
        result.append({"component_indices":component["indices"],"covariance_count":len(summaries),
                       "names":names,"signed_mean":average(mean).tolist(),
                       "signed_mean_in_natural_units":(average(mean)/scale).tolist(),
                       "equal_covariance_flag_fraction":flag_fraction.tolist(),
                       "equal_covariance_positive_fraction":positive_fraction.tolist(),
                       "persistent_same_sign_residual_flags":persistent.tolist(),
                       "persistent_names":[name for name,flagged in zip(names,persistent)if flagged],
                       "qualification":"persistent residuals oppose agreement; correlated nodes/xi do not provide independent replicates"})
    return result


def graph_only_tolerance(calibration, *, normalization="mean", weighting="primary"):
    reference=calibration["results"][f"grounded:{normalization}"]
    cohorts=[row for name,row in reference["cohorts"].items()if name.startswith("graph_")]
    if not cohorts:raise ValueError("No graph reference cohorts in calibration")
    return float(max(row["epsilon_by_weight"][weighting]for row in cohorts))


def direct_graph_levelset_comparisons(nodes,graph_features,calibration,*,epsilon,
                                     normalization="mean",weighting="primary",
                                     covariance_separation=.35,bootstrap_vectors=None,
                                     bootstrap_valid=None):
    """Directly compare graphs assigned a common measured xi in low-J regions.

    Every graph pair is listed once with all shared xi values. Point scores use
    natural units; the graph-only old-data reference is retained separately.
    Optional bootstrap arrays map ID->draws,F, resampled by physical parents.
    They cannot establish equality, and selected near-matches are not null tests.
    """
    grouped=defaultdict(dict)
    for node in nodes:
        if node.get("eligible",False)and node["score"]<=epsilon:
            grouped[float(node["xi"])][str(node["id"])]=node
    pair_xi=defaultdict(list);positions={}
    for xi,rows in grouped.items():
        for identifier,node in rows.items():positions[identifier]=np.asarray(node["triple"],float)
        for a,b in combinations(sorted(rows),2):
            if np.linalg.norm(positions[a]-positions[b])>=covariance_separation:
                pair_xi[(a,b)].append(xi)
    tolerance=graph_only_tolerance(calibration,normalization=normalization,weighting=weighting)
    result=[]
    for (a,b),xis in sorted(pair_xi.items()):
        left,right=graph_features[a],graph_features[b]
        point=grouped_feature_distance(left,right,weights=WEIGHTINGS[weighting])
        delta,valid=_valid_delta(left,right)
        row={"left":a,"right":b,"shared_observed_xi":sorted(xis),
             "covariance_separation":float(np.linalg.norm(positions[a]-positions[b])),
             "score":point["total"],"components":point["components"],
             "graph_only_reference_epsilon":tolerance,
             "point_within_reference_noise":bool(point["total"]<=tolerance),
             "signed_residuals":delta.tolist(),"valid":valid.tolist()}
        if bootstrap_vectors is not None and a in bootstrap_vectors and b in bootstrap_vectors:
            va,vb=np.asarray(bootstrap_vectors[a]),np.asarray(bootstrap_vectors[b])
            if va.shape!=vb.shape or va.ndim!=2:raise ValueError("Graph bootstrap shapes must match draws,F")
            ma=np.ones_like(va,dtype=bool)if bootstrap_valid is None else np.asarray(bootstrap_valid[a],bool)
            mb=np.ones_like(vb,dtype=bool)if bootstrap_valid is None else np.asarray(bootstrap_valid[b],bool)
            scores=[]
            for x,y,mx,my in zip(va,vb,ma,mb):
                fa=dict(left,vector=x,valid=mx);fb=dict(right,vector=y,valid=my)
                scores.append(grouped_feature_distance(fa,fb,weights=WEIGHTINGS[weighting])["total"])
            row["mc95"]=np.quantile(scores,[.025,.975]).tolist()
            row["bootstrap_draws"]=len(scores)
        result.append(row)
    return {"rows":result,"pairs":len(result),"minimum_covariance_separation":covariance_separation,
            "graph_only_reference_epsilon":tolerance,
            "qualification":"direct finite graph comparison; common xi assignment and low scores do not prove equal laws; selection and systematic errors excluded"}


def propose_family_adaptation(nodes,existing,*,epsilon,count=12,first_identifier=None,
                               seed=202611123501,minimum_eigenvalue=.15,condition_limit=12.):
    """Spread measured-data-driven proposals over xi, uncertainty and coverage.

    Allocation:4 observed-edge/branch probes,3 threshold-uncertainty probes,
    3 signed-component-disagreement probes,2 maximin coverage probes. Proposals
    are not evidence of connections; no interpolated score is assigned to them.
    Fallbacks and the actual allocation are saved if a category has no support.
    """
    from src.covariance_design_3b import configuration,canonical_triple
    from scipy.stats import qmc

    if count<1:raise ValueError("Positive adaptation count required")
    nodes=list(nodes);existing=list(existing);rng=np.random.default_rng(seed)
    coordinates=np.array([row["triple"]for row in existing],float)
    identifiers=[int(str(row["id"])[1:])for row in existing]
    first=max(identifiers)+1 if first_identifier is None else int(first_identifier)
    available=[node for node in nodes if node.get("eligible",False)and np.isfinite(node["score"])and node["xi"]>0]
    if not available:raise ValueError("No eligible observed nodes for adaptation")
    low=[node for node in available if node["score"]<=epsilon]
    primary=observed_family_components(nodes,epsilon=epsilon,tested_covariances=existing)
    edges=[(nodes[a],nodes[b])for a,b in primary["edges"]if nodes[a]["id"]!=nodes[b]["id"]]
    chosen=[];used_anchors=[];attempts=[]

    def feasible(triple,reason,metadata):
        try:spec=configuration(canonical_triple(triple),family=f"phase3c_{reason}")
        except ValueError:return None
        if spec["condition"]>condition_limit or spec["eigenvalues"][0]<minimum_eigenvalue:return None
        reference=coordinates if not chosen else np.vstack([coordinates,[r["triple"]for r in chosen]])
        if np.linalg.norm(reference-np.asarray(spec["triple"]),axis=1).min()<.035:return None
        spec["acquisition"]={"category":reason,**metadata,
                              "predicted_score":None,"status":"proposal only; new measured paths required"}
        return spec

    def jitter(anchor,reason):
        pool=[]
        for _ in range(80):
            direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
            spec=feasible(np.asarray(anchor["triple"])+rng.uniform(.055,.12)*direction,
                          reason,{"anchor_id":anchor["id"],"target_xi":float(anchor["xi"]),
                                  "observed_anchor_score":float(anchor["score"])})
            if spec is not None:pool.append(spec)
        if not pool:return None
        reference=coordinates if not chosen else np.vstack([coordinates,[r["triple"]for r in chosen]])
        distance=[np.linalg.norm(reference-np.asarray(row["triple"]),axis=1).min()for row in pool]
        return pool[int(np.argmax(distance))]

    targets=[.125,.175,.225,.275]
    for target in targets[:min(4,count)]:
        candidates=sorted(edges,key=lambda pair:abs(pair[0]["xi"]-target))
        proposals=[]
        for a,b in candidates:
            if abs(a["xi"]-target)>min(abs(n["xi"]-target)for n in available)+.025:continue
            spec=feasible((np.asarray(a["triple"])+b["triple"])/2,"observed_edge_probe",
                          {"anchor_ids":[a["id"],b["id"]],"target_xi":float(a["xi"]),
                           "question":"test an unobserved midpoint between measured neighboring low-J nodes"})
            if spec is not None:proposals.append(spec)
        if proposals:
            reference=coordinates if not chosen else np.vstack([coordinates,[r["triple"]for r in chosen]])
            choice=max(proposals,key=lambda row:np.linalg.norm(reference-np.asarray(row["triple"]),axis=1).min())
        else:
            anchors=low or available
            near=min(abs(node["xi"]-target)for node in anchors)
            pool=[node for node in anchors if abs(node["xi"]-target)<=near+1e-12]
            anchor=max(pool,key=lambda node:min(np.linalg.norm(np.asarray(node["triple"])-x)for x in used_anchors)
                       if used_anchors else np.linalg.norm(node["triple"]))
            choice=jitter(anchor,"branch_coverage_fallback")
        if choice is not None:chosen.append(choice);used_anchors.append(np.asarray(choice["triple"]))
        attempts.append({"category":"branch_continuity","target_xi":target,"success":choice is not None})
    for category,targets in [("threshold_uncertainty",[.15,.2125,.2875]),
                             ("component_disagreement",[.175,.2375,.3])]:
        for target in targets:
            if len(chosen)>=min(count,10):break
            near=min(abs(node["xi"]-target)for node in available)
            pool=[node for node in available if abs(node["xi"]-target)<=near+1e-12]
            if category=="threshold_uncertainty":
                def priority(node):
                    ci=node.get("robust_mc95",node.get("score_mc95"))
                    width=(ci[1]-ci[0])if ci is not None else .02
                    return width/(abs(node["score"]-epsilon)+.02)
            else:
                def priority(node):
                    flags=node.get("family_flags",{})
                    return sum(bool(v)for v in flags.values())+.1*float(np.linalg.norm(node.get("signed_residuals",[])))
            ordered=sorted(pool,key=priority,reverse=True)
            choice=None
            for anchor in ordered:
                if any(np.linalg.norm(np.asarray(anchor["triple"])-x) < .10 for x in used_anchors):continue
                choice=jitter(anchor,category)
                if choice is not None:break
            if choice is not None:chosen.append(choice);used_anchors.append(np.asarray(choice["triple"]))
            attempts.append({"category":category,"target_xi":target,"success":choice is not None})
    cloud=qmc.scale(qmc.LatinHypercube(3,seed=seed+1).random(512),[-.75]*3,[.75]*3)
    while len(chosen)<count:
        pool=[spec for triple in cloud if(spec:=feasible(triple,"maximin_coverage",
                    {"target_xi":None,"question":"cover covariance directions away from existing family/argmin anchors"}))is not None]
        if not pool:break
        reference=np.vstack([coordinates,[r["triple"]for r in chosen]])if chosen else coordinates
        choice=max(pool,key=lambda row:np.linalg.norm(reference-np.asarray(row["triple"]),axis=1).min())
        chosen.append(choice)
    for index,row in enumerate(chosen):row["id"]=f"C{first+index:03d}"
    return {"configurations":chosen,"seed":seed,"attempted_categories":attempts,
            "actual_category_counts":dict((category,sum(r["acquisition"]["category"]==category for r in chosen))
                         for category in sorted({r["acquisition"]["category"]for r in chosen})),
            "no_global_argmin_optimization":True,"minimum_eigenvalue":minimum_eigenvalue,
            "condition_limit":condition_limit,"minimum_existing_distance":.035,
            "warning":"xi/edge information only guides proposals; new samples, gates and held-out/refinement evidence required"}


def local_feature_rank(covariances,vectors,center_index,scales,*,radius=.25,max_neighbors=12,
                       valid=None,bootstrap_vectors=None,field_tangent=None,
                       bootstrap_field_tangents=None,quotient_wall_tolerance=.03):
    """Local linear feature response with whole-parent bootstrap diagnostics.

    Multiple GLOBAL PCs do not establish local rank. Even a local response is
    affected by finite radius curvature, permutation-quotient walls, small design
    rank and covariance-dependent sampler bias. Bootstrap evidence is descriptive.
    """
    c=np.asarray(covariances,float);y=np.asarray(vectors,float);scale=np.asarray(scales,float)
    if c.ndim!=2 or c.shape[1]!=3 or y.ndim!=2 or len(c)!=len(y)or scale.shape!=(y.shape[1],):
        raise ValueError("Need C[N,3],features[N,F],scales[F]")
    if np.any(scale<=0)or np.any(~np.isfinite(scale)):
        raise ValueError("Positive finite natural feature scales required")
    distance=np.linalg.norm(c-c[center_index],axis=1)
    neighbors=np.argsort(distance)
    neighbors=neighbors[distance[neighbors]<=radius+1e-12][:max_neighbors]
    if len(neighbors)<5:
        return {"status":"insufficient_neighbors","neighbors":neighbors.tolist()}
    mask=np.all(np.isfinite(y[neighbors]),axis=0)
    if valid is not None:mask&=np.all(np.asarray(valid,bool)[neighbors],axis=0)
    if not np.any(mask):return {"status":"no_common_valid_features","neighbors":neighbors.tolist()}
    design=np.column_stack((np.ones(len(neighbors)),c[neighbors]-c[center_index]))
    if np.linalg.matrix_rank(design)<4:
        return {"status":"insufficient_covariance_directions","neighbors":neighbors.tolist()}
    weight=np.exp(-2*(distance[neighbors]/radius)**2)
    solver=np.linalg.pinv(np.sqrt(weight)[:,None]*design)*np.sqrt(weight)[None,:]
    coefficients=solver@(y[neighbors][:,mask]/scale[mask])
    derivative=coefficients[1:]
    singular=np.linalg.svd(derivative,compute_uv=False)
    wall=bool(np.min(np.diff(np.sort(c[center_index])))<quotient_wall_tolerance)
    result={"status":"descriptive_local_fit","neighbors":neighbors.tolist(),"radius":radius,
            "design_condition":float(np.linalg.cond(np.sqrt(weight)[:,None]*design)),
            "singular_values":singular.tolist(),"jacobian":derivative.tolist(),
            "valid_feature_indices":np.flatnonzero(mask).tolist(),"near_permutation_quotient_wall":wall,
            "warning":"no continuum rank claim; smooth covariance-dependent sampler bias can mimic a local derivative"}
    if field_tangent is not None:
        tangent=np.asarray(field_tangent,float)[mask]/scale[mask]
        norm=float(tangent@tangent)
        if norm>1e-14:
            transverse=derivative-np.outer(derivative@tangent,tangent)/norm
            result["transverse_to_field_tangent_singular_values"]=np.linalg.svd(transverse,compute_uv=False).tolist()
            result["field_tangent_uncertainty_included"]=False
    if bootstrap_vectors is not None:
        bootstrap=np.asarray(bootstrap_vectors,float)
        if bootstrap.ndim!=3 or bootstrap.shape[1:]!=y.shape:
            raise ValueError("bootstrap_vectors must have shape draws,N,F; each draw must resample whole parents")
        derivatives=np.array([(solver@(draw[neighbors][:,mask]/scale[mask]))[1:]for draw in bootstrap])
        noise=np.array([np.linalg.norm(d-derivative,ord=2)for d in derivatives])
        bound=float(np.quantile(noise,.95))
        result["conditional_bootstrap_operator_noise95"]=bound
        numerical_floor=100*np.finfo(float).eps*max(float(singular[0]),1.)
        result["singular_directions_above_conditional_noise"]=int(np.sum(singular>max(bound,numerical_floor)))
        result["bootstrap_singular_mc95"]=np.quantile([np.linalg.svd(d,compute_uv=False)for d in derivatives],[.025,.975],axis=0).tolist()
        result["noise_qualification"]="conditional whole-parent bootstrap only; excludes curvature, numerical biases and local-design selection"
        if field_tangent is not None and bootstrap_field_tangents is not None and norm>1e-14:
            tangent_draws=np.asarray(bootstrap_field_tangents,float)
            if tangent_draws.shape!=(len(derivatives),y.shape[1]):
                raise ValueError("bootstrap_field_tangents must match draws,F")
            perpendicular=[]
            for derivative_draw,t in zip(derivatives,tangent_draws[:,mask]/scale[mask]):
                t_norm=float(t@t)
                if t_norm<=1e-14:continue
                perpendicular.append(derivative_draw-np.outer(derivative_draw@t,t)/t_norm)
            if perpendicular:
                tangent_noise95=float(np.quantile([np.linalg.norm(d-transverse,ord=2)for d in perpendicular],.95))
                result["transverse_graph_and_field_bootstrap_noise95"]=tangent_noise95
                result["field_tangent_uncertainty_included"]=True
    return result
