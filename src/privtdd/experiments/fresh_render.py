"""Render new run results, preserving their own completion coverage.

These figures describe the new execution. They are never labeled as the frozen
paper-v1 numbers, even when the design is the same.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .render.e1_statistics import interval


def sampling_with_e1_controls(sample, releases, *, smoke=False):
    """Use every completed E1 control for production accuracy summaries.

    The sampling workers use five paired controls for timing/comparison. The
    paper's full-collection accuracy row uses the entire E1 repetition set.
    Smoke sampling has its own graph/control and is intentionally kept separate.
    """
    sample=sample.copy()
    sample['accuracy_source']='sampling execution'
    if smoke:
        return sample
    cases=sample[['dataset','epsilon']].drop_duplicates()
    full=releases[(releases.panel=='main') & (releases.method=='PrivTDD')].copy()
    full=full.merge(cases,on=['dataset','epsilon'],how='inner',validate='many_to_one')
    if full.duplicated(['dataset','epsilon','seed']).any():
        raise ValueError('Duplicate E1 controls in fresh sampling summary')
    full['rate']=1.
    full['status']='complete'
    full['accuracy_source']='all completed E1 PrivTDD controls'
    return pd.concat([sample[sample.rate<1],full],ignore_index=True,sort=False)


def render_fresh_runs(run_root, output_dir=None, sampling_root=None):
    root=Path(run_root).resolve()
    out=Path(output_dir or root/'rendered').resolve()
    if 'published' in out.parts:
        raise ValueError('Fresh results cannot overwrite a published snapshot')
    out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((root/'forpaper/plan.json').read_bytes())
    records=[];coverage=[]
    for task in plan['jobs']:
        folder=root/'forpaper/jobs'/task['id']
        status=json.loads((folder/'status.json').read_bytes()) if (folder/'status.json').exists() else {'status':'pending'}
        coverage.append(dict(job_id=task['id'],panel=task['panel'],dataset=task['dataset'],epsilon=task['epsilon'],seed=task['seed'],status=status['status']))
        if status['status']=='complete':
            from .common import validate_completion
            validate_completion(folder,status)
            for row in json.loads((folder/'records.json').read_bytes()):
                row=dict(row,job_id=task['id'],EMD=row['W1'])
                row['method']=row.get('canonical_method',row['method'])
                records.append(row)
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    if not records:
        raise ValueError('No completed outputs to render; coverage.csv records the pending/failed tasks')
    frame=pd.DataFrame(records)
    frame.to_csv(out/'releases.csv',index=False)
    stats=[]
    for keys,group in frame.groupby(['panel','dataset','epsilon','method']):
        group=group.sort_values('seed')
        if group.seed.duplicated().any():raise ValueError(f'Duplicate releases at {keys}')
        for metric in ('MSE','EMD','KS'):
            stats.append(dict(zip(('panel','dataset','epsilon','method'),keys),metric=metric,**interval(group[metric])))
    summary=pd.DataFrame(stats)
    summary.to_csv(out/'summary.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    generated=[]
    for panel,part in summary.groupby('panel'):
        networks=sorted(part.dataset.unique())
        fig,axes=plt.subplots(len(networks),3,figsize=(10,2.6*len(networks)),squeeze=False,layout='constrained')
        for i,dataset in enumerate(networks):
            for j,metric in enumerate(('MSE','EMD','KS')):
                axis=axes[i,j]
                for method,series in part[(part.dataset==dataset)&(part.metric==metric)].groupby('method'):
                    series=series.sort_values('epsilon')
                    axis.plot(series.epsilon,series['mean'],marker='o',label=method)
                    if series.lower95.notna().all():
                        axis.fill_between(series.epsilon.to_numpy(),series.lower95.to_numpy(),series.upper95.to_numpy(),alpha=.12)
                axis.set(xlabel='Privacy budget',ylabel=metric,title=dataset)
                if metric!='KS' and (part[part.metric==metric]['mean']>0).all():axis.set_yscale('log')
                axis.grid(alpha=.18)
                if i==0 and j==0:axis.legend(fontsize=7)
        fig.suptitle(f'New execution: {panel}; means over completed runs')
        file=out/f'{panel}.pdf';fig.savefig(file);plt.close(fig);generated.append(file.name)
    if sampling_root is not None:
        sample_path=Path(sampling_root)/'summary/runs.csv'
        if sample_path.exists():
            sample=pd.read_csv(sample_path,float_precision='round_trip')
            sample.to_csv(out/'sampling_worker_releases.csv',index=False)
            sample=sampling_with_e1_controls(sample,frame,smoke=plan['smoke'])
            sample.to_csv(out/'sampling_releases.csv',index=False)
            if 'status' in sample:sample=sample[sample.status=='complete']
            sampled_stats=[]
            for keys,group in sample.groupby(['dataset','epsilon','rate']):
                for metric in ('MSE','EMD','KS'):
                    sampled_stats.append(dict(zip(('dataset','epsilon','rate'),keys),metric=metric,**interval(group.sort_values('seed')[metric])))
            pd.DataFrame(sampled_stats).to_csv(out/'sampling_summary.csv',index=False)
    report=dict(kind='fresh execution, not paper-v1',planned_jobs=len(coverage),completed_jobs=sum(c['status']=='complete' for c in coverage),
                method_outputs=len(frame),generated_figures=generated,code=plan['code'],smoke=plan['smoke'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report
