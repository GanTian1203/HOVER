"""Manuscript-only views of accepted evidence; no training or GPU entry point."""
import argparse
import json
from pathlib import Path

from .io import config, load, save, sha256


def panel_plan(panel):
    policy = config('paper_presentation')
    spec = policy['panels'][panel]
    systems = spec.get('systems') or [b+'+'+a for b in spec['backbones'] for a in spec['arms']]
    rows = []
    for dataset in spec['datasets']:
        horizons = spec.get('horizons_by_dataset', {}).get(dataset, spec.get('horizons', []))
        for horizon in horizons:
            for system in systems:
                for seed in policy['seeds']:
                    rows.append(dict(dataset=dataset,horizon=horizon,system=system,seed=seed,role='DEV'))
    if len(rows) != spec['records']:
        raise ValueError('Presentation configuration count mismatch')
    return dict(status='PRESENTATION_PLAN_ONLY',version=policy['version'],panel=panel,tier=spec['tier'],
                record_count=len(rows),reference=spec['reference'],rows=rows,
                new_fits_authorized=0,GPU_execution=False,TEST_access=False)


def reference_view(panel, reference_root):
    spec = config('paper_presentation')['panels'][panel]
    reference_root = Path(reference_root)
    if spec.get('requires_complete_C_acceptance'):
        provenance = load(reference_root/'accepted_reference_provenance.json')
        if provenance.get('C_final') is not True or provenance.get('C_records') != 240:
            raise ValueError('C final paper views require complete independent PASS_240 evidence; no partial final table or placeholder is allowed')
    source = reference_root/spec['reference']
    if spec['format'] == 'json':
        records = load(source)
    else:
        from .analysis import records_from_csv
        records = records_from_csv(source,'ablation' if spec['format']=='ablation_csv' else 'main')
    plan = panel_plan(panel)
    key = lambda r: (r['dataset'],r['horizon'],r['system'],r['seed'])
    wanted = {key(r) for r in plan['rows']}
    selected = [r for r in records if key(r) in wanted]
    if len(selected)!=len(wanted) or {key(r) for r in selected}!=wanted:
        raise ValueError('Missing or duplicated five-seed evidence for the fixed presentation panel')
    if any(r.get('role')!='DEV' or r.get('TEST_access') is not False or r.get('status')!='DONE' for r in selected):
        raise ValueError('Unaccepted evaluation role/status')
    if panel in ('Table2','Capacity','C'):
        paired = {}
        for row in selected:
            backbone = row['system'].split('+')[0] if '+' in row['system'] else 'OLINEAR'
            paired.setdefault((row['dataset'],row['horizon'],backbone,row['seed']),set()).add(row['base_checkpoint_SHA256'])
        if any(len(values)!=1 or not next(iter(values)) for values in paired.values()):
            raise ValueError('Same-seed frozen backbone identity mismatch')
    return selected, dict(version=plan['version'],panel=panel,record_count=len(selected),source_SHA256=sha256(source),
                          kind='EXISTING_ACCEPTED_DEV_VIEW',new_fits_authorized=0,TEST_access=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',choices=('A','Table2','Capacity','B','C','all'),default='all')
    parser.add_argument('--output',help='Optional presentation-plan JSON')
    parser.add_argument('--reference-root',default='results/reference')
    parser.add_argument('--export-records',help='Optional JSON view of existing accepted records; one panel only')
    args=parser.parse_args(argv)
    try:
        def write_view(path,value):
            path=Path(path).resolve()
            reference_root=Path(args.reference_root).resolve()
            if path==reference_root or reference_root in path.parents:
                raise ValueError('Presentation exports cannot overwrite the accepted reference directory')
            if path.exists():
                if load(path)==value:return
                raise FileExistsError('Existing output differs; preserve it and choose a new view path')
            save(path,value)
        if args.export_records:
            if args.panel=='all':raise ValueError('Export one panel at a time')
            records,result=reference_view(args.panel,args.reference_root)
            write_view(args.export_records,records)
        elif args.panel=='all':
            result=dict(status='PRESENTATION_PLANS_ONLY',panels={p:panel_plan(p) for p in config('paper_presentation')['panels']},new_fits_authorized=0,TEST_access=False)
        else:result=panel_plan(args.panel)
        if args.output:write_view(args.output,result)
        print(json.dumps(result,indent=2))
        return 0
    except (ValueError,OSError,KeyError) as error:
        print(json.dumps(dict(status='BLOCKED_REFERENCE_VIEW',error=str(error),new_fits_authorized=0,TEST_access=False)))
        return 2


if __name__=='__main__':raise SystemExit(main())
