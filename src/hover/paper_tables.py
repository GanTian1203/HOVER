"""Recompute accepted paper statistics without training or dataset access."""
import argparse,json
from pathlib import Path
from .analysis import records_from_csv,summarize,placements,pair,write_csv
from .io import load,save

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',default='results/reference');p.add_argument('--output',default='outputs/paper_tables')
    a=p.parse_args();root=Path(a.reference_root);out=Path(a.output)
    if out.resolve()==root.resolve() or root.resolve() in out.resolve().parents:raise ValueError('Never overwrite accepted references')
    A=records_from_csv(root/'main_DEV_seeds.csv');B=records_from_csv(root/'ablation_DEV_seeds.csv','ablation')
    T=load(root/'cross_backbone_DEV_records.json');C=load(root/'horizon_accepted_DEV_records.json')
    assert len(A)==600 and len(B)==300 and len(T)==225 and len(C)==240
    for name,rs in [('Table1',A),('Table2_three_arm',T),('Table3',B),('Table4_three_arm',C)]:write_csv(out/(name+'_MSE_MAE_mean_SD.csv'),summarize(rs))
    _,counts=placements(summarize(A));write_csv(out/'Table1_metric_best_count.csv',counts)
    pairs=[]
    for backbone in ['NLINEAR_PAIRED','DLINEAR','FITS','ITRANSFORMER','OLINEAR']:
        for ref in ['FROZEN','SIMPLE']:
            _,s=pair(T,backbone+'+HOVER',backbone+'+'+ref)
            pairs.extend(dict(backbone=backbone,**r) for r in s)
    write_csv(out/'Table2_all_same_backbone_pairs.csv',pairs)
    controls=[]
    for control in ['history_only','no_history_direct','concatenation','history_condition_replacement']:
        _,s=pair(B,'HOVER',control);controls.extend(s)
    write_csv(out/'Table3_all_control_pairs.csv',controls)
    cs=[]
    for ref in ['FROZEN','SIMPLE']:
        _,s=pair(C,'HOVER',ref);cs.extend(s)
    write_csv(out/'Table4_all_pairs.csv',cs)
    summary=dict(status='ACCEPTED_DEV_TABLES_RECOMPUTED',records=dict(A=600,Table2=225,B_controls_and_full=300,C=240),new_fits=0,TEST_access=False)
    save(out/'verification.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
