"""Independently recompute frozen metrics from the standalone local archive.

Read-only artifact access; reports are written only under --output.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--rtol',type=float,default=1e-10)
parser.add_argument('--atol',type=float,default=1e-10)
args=parser.parse_args()
root=args.root.resolve();out=args.output.resolve()
if out.is_relative_to(root/'src') or out.is_relative_to(root/'results/published') or out.is_relative_to(root/'results/archive') or out.is_relative_to(root/'datasets'):
    raise ValueError('Verification output must be separate from source and immutable evidence')
out.mkdir(parents=True,exist_ok=True)
sys.dont_write_bytecode=True
sys.path.insert(0,str(root/'src'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from privtdd.evaluation import metrics,e3_diagnostics
from privtdd.io.identity import source_digest
from privtdd.experiments.reproduce import verify_snapshot

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

started=time.perf_counter()
source_before=source_digest('package')
report=dict(root=str(root),source_digest_before=source_before,
    generator=str(Path(__file__).name),generator_sha256=sha(__file__),
    metric_implementation=str(Path(metrics.__file__).relative_to(root)),metric_implementation_sha256=sha(metrics.__file__),
    tolerance=dict(rtol=args.rtol,atol=args.atol,rule='abs(recomputed - frozen) <= atol + rtol * abs(frozen)'),
    method='Load SHA-verified archived keys/counts and processed per-node triangle degrees; form the exact truth histogram and call privtdd.evaluation.metrics.compare with n and public degree cap n-1. Deduplicate arithmetic by (dataset,prediction SHA); compare every referenced frozen CSV row.')
report['frozen_snapshot']=verify_snapshot(root)
reference_path=root/'provenance/prediction_references.csv'
refs=csv_rows(reference_path)
report['reference_manifest_sha256']=sha(reference_path)
report['references']=len(refs)
report['unique_prediction_files']=len(set(r['prediction_sha256'] for r in refs))
mapping={'Bio-DMELA':'bio-dmela','DBLP':'dblp','Higgs social':'higgs','Orkut':'orkut','higgs':'higgs','orkut':'orkut'}
failures=[];truth={};truth_info={};tables={};verified_blobs=set();cache={};unique_rows=[];comparisons=[]
maxima={name:dict(absolute_error=0.,relative_error=0.,absolute_error_reference=None) for name in ['MSE','EMD','KS']}
for ds in sorted(set(mapping[r['dataset']] for r in refs)):
    folder=root/'datasets/processed'/ds
    graph=json.loads((folder/'graph.json').read_text(encoding='utf-8'))
    values=np.load(folder/'triangles.npy',allow_pickle=False)
    assert values.ndim==1 and len(values)==graph['nodes'] and np.isfinite(values).all()
    assert np.issubdtype(values.dtype,np.integer) and np.all(values>=0)
    assert graph['public_degree_cap']==len(values)-1
    keys,counts=np.unique(values,return_counts=True)
    truth[ds]=(keys,counts,graph['nodes'],graph['public_degree_cap'])
    truth_info[ds]=dict(nodes=len(values),support_points=len(keys),public_degree_cap=graph['public_degree_cap'],
        truth_sha256=sha(folder/'triangles.npy'),graph_metadata_sha256=sha(folder/'graph.json'),
        triangle_degree_sum=int(values.sum(dtype=np.int64)))
report['truth_inputs']=truth_info
for name in sorted(set(r['table'] for r in refs)):
    file=root/'results/published/paper-v1'/name
    rows=csv_rows(file);tables[name]=rows
    matched=[r for r in refs if r['table']==name]
    if sorted(int(r['csv_line']) for r in matched)!=list(range(2,len(rows)+2)):
        failures.append(dict(kind='reference_coverage',table=name,table_rows=len(rows),references=len(matched)))
report['table_coverage']={name:dict(rows=len(rows),references=sum(r['table']==name for r in refs),
    sha256=sha(root/'results/published/paper-v1'/name)) for name,rows in tables.items()}

with threadpool_limits(limits=1):
    for index,ref in enumerate(refs):
        identity=dict(table=ref['table'],csv_line=int(ref['csv_line']),prediction_sha256=ref['prediction_sha256'])
        try:
            frozen=tables[ref['table']][int(ref['csv_line'])-2]
            for column in ('prediction_sha256','dataset','epsilon','seed','report_sha256','truth_sha256','rate'):
                actual=frozen.get(column,'');expected=ref.get(column,'')
                equal=(float(actual)==float(expected)) if column in ('epsilon','seed','rate') and actual and expected else actual==expected
                if not equal:raise ValueError(f'Reference identity mismatch {column}: {actual!r} != {expected!r}')
            if frozen.get('method',frozen.get('quantiles',''))!=ref['method']:
                raise ValueError('Reference method/quantile identity mismatch')
            ds=mapping[ref['dataset']]
            if ref['truth_sha256']!=truth_info[ds]['truth_sha256']:
                raise ValueError('Reference truth checksum differs from processed data')
            prediction_sha=ref['prediction_sha256'];key=(ds,prediction_sha)
            if key not in cache:
                file=root/'results/archive/predictions'/f'{prediction_sha}.npz'
                raw=file.read_bytes()
                if hashlib.sha256(raw).hexdigest()!=prediction_sha:raise ValueError('Prediction content checksum mismatch')
                verified_blobs.add(prediction_sha)
                with np.load(io.BytesIO(raw),allow_pickle=False) as saved:
                    keys,counts=saved['keys'],saved['counts']
                true_keys,true_counts,n,cap=truth[ds]
                result=metrics.compare(true_keys,true_counts,keys,counts,n=n,public_cap=cap)
                cache[key]=dict(MSE=result['MSE'],EMD=result['W1'],KS=result['KS'])
                unique_rows.append(dict(dataset=ds,prediction_sha256=prediction_sha,truth_sha256=truth_info[ds]['truth_sha256'],
                    n=n,public_degree_cap=cap,**cache[key]))
            actual=cache[key];comparison=dict(**identity,dataset=ds,method=ref['method'],seed=int(ref['seed']))
            for name in ('MSE','EMD','KS'):
                expected=float(frozen[name]);value=actual[name]
                if not np.isfinite(expected) or not np.isfinite(value):raise ValueError(f'Non-finite metric {name}')
                difference=abs(value-expected);relative=difference/abs(expected) if expected else (0. if not difference else None)
                passes=bool(np.isclose(value,expected,rtol=args.rtol,atol=args.atol))
                comparison.update({name+'_frozen':expected,name+'_recomputed':value,name+'_absolute_error':difference,name+'_passed':passes})
                if difference>maxima[name]['absolute_error']:
                    maxima[name].update(absolute_error=difference,absolute_error_reference=identity)
                if relative is not None:maxima[name]['relative_error']=max(maxima[name]['relative_error'],relative)
                if not passes:failures.append(dict(kind='metric',metric=name,expected=expected,recomputed=value,
                                                  absolute_error=difference,**identity))
            comparisons.append(comparison)
        except Exception as error:
            failures.append(dict(kind='row_exception',error=f'{type(error).__name__}: {error}',**identity))
        if (index+1)%500==0 or index+1==len(refs):
            print(f'Checked {index+1}/{len(refs)} references; {len(cache)} unique evaluations; failures={len(failures)}; elapsed={time.perf_counter()-started:.1f}s',flush=True)

pd.DataFrame(unique_rows).to_csv(out/'unique_prediction_metrics.csv',index=False)
pd.DataFrame(comparisons).to_csv(out/'row_metric_comparison.csv',index=False)
report.update(unique_evaluations=len(cache),sha_verified_prediction_files=len(verified_blobs),
    referenced_rows_evaluated=len(comparisons),metric_comparisons=3*len(comparisons),maximum_errors=maxima,
    per_table={name:dict(compared_rows=sum(r['table']==name for r in comparisons),
                         failed_entries=sum(r.get('table')==name for r in failures)) for name in tables})

try:
    diagnostic_manifest=root/'provenance/e3_diagnostic_inputs.json'
    diag=e3_diagnostics.from_manifest(diagnostic_manifest,root,out/'e3_regenerated')
    frozen=pd.read_csv(root/'results/published/paper-v1/e3/prose_regions.csv',float_precision='round_trip')
    regenerated=pd.read_csv(out/'e3_regenerated/prose_regions.csv',float_precision='round_trip')
    identifiers=['dataset','seed','method']
    frozen=frozen.sort_values(identifiers).reset_index(drop=True)
    regenerated=regenerated.sort_values(identifiers).reset_index(drop=True)
    if list(frozen.columns)!=list(regenerated.columns) or len(frozen)!=len(regenerated):
        raise ValueError('Regenerated E3 prose row or column coverage differs')
    numeric_checks=0;diag_max=0.
    for col in frozen.columns:
        if pd.api.types.is_numeric_dtype(frozen[col]):
            a=frozen[col].to_numpy();b=regenerated[col].to_numpy()
            np.testing.assert_allclose(b,a,rtol=args.rtol,atol=args.atol)
            numeric_checks+=len(a);diag_max=max(diag_max,float(np.max(np.abs(b-a))))
        elif not frozen[col].equals(regenerated[col]):raise ValueError(f'E3 identity field mismatch: {col}')
    report['e3_diagnostics']=dict(status='passed',manifest_sha256=sha(diagnostic_manifest),
        predictions=len(regenerated),numeric_values_compared=numeric_checks,max_absolute_error=diag_max,
        derived_report=diag,output='e3_regenerated')
except Exception as error:
    failures.append(dict(kind='e3_diagnostics',error=f'{type(error).__name__}: {error}'))
    report['e3_diagnostics']=dict(status='failed',error=str(error))

report['source_digest_after']=source_digest('package')
if report['source_digest_after']!=source_before:failures.append(dict(kind='source_changed_during_check'))
report['elapsed_seconds']=time.perf_counter()-started
report['failures']=failures
report['failure_count']=len(failures)
report['status']='passed' if not failures else 'failed'
report['outputs']={name:dict(sha256=sha(out/name),bytes=(out/name).stat().st_size)
                   for name in ['unique_prediction_metrics.csv','row_metric_comparison.csv']}
write_json(out/'archive_numeric_verification.json',report)
print(json.dumps({k:report[k] for k in ('status','references','unique_evaluations','metric_comparisons','maximum_errors','failure_count','elapsed_seconds')},indent=2),flush=True)
sys.exit(1 if failures else 0)
