"""Rebuild the publication summaries and paired intervals from releases.csv."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
KEY=["dataset","epsilon","seed"]
METHODS={"E2":["PrivTDD","SCRec","Fixed-0.25","Fixed-0.50","Fixed-0.75"],
         "E3":["(0.80,0.90)","(0.90,0.95)","(0.95,0.99)"]}


def summarize(x):
    values=np.asarray(x,dtype=float)
    assert len(values)>1 and np.isfinite(values).all()
    samples=np.random.default_rng(731).integers(len(values),size=(2000,len(values)))
    lo,hi=np.quantile(values[samples].mean(axis=1),[.025,.975])
    return dict(n=len(values),mean=values.mean(),sd=values.std(ddof=1),lower95=lo,upper95=hi)


def main(folder=None):
    global HERE
    if folder is not None: HERE = Path(folder)
    rows=pd.read_csv(HERE/"releases.csv", float_precision="round_trip")
    means=[]; differences=[]
    for suite in METHODS:
        data=rows[rows.suite==suite]
        assert not data.duplicated(KEY+["method"]).any()
        assert data.groupby(KEY).report_sha256.nunique().eq(1).all()
        assert data.groupby(KEY).truth_sha256.nunique().eq(1).all()
        ref="PrivTDD" if suite=="E2" else "(0.90,0.95)"
        for (ds,eps), group in data.groupby(KEY[:2]):
            for metric in ["MSE","EMD","KS"]:
                table=group.pivot(index="seed",columns="method",values=metric).sort_index()
                assert table.notna().all().all() and set(table.columns)==set(METHODS[suite])
                for name in METHODS[suite]:
                    key=dict(suite=suite,dataset=ds,epsilon=eps,method=name,metric=metric)
                    means.append(dict(**key,**summarize(table[name])))
                    if name!=ref:
                        ratio=table[name].mean()/table[ref].mean()
                        differences.append(dict(**key,reference=ref,**summarize(table[name]-table[ref]),
                                                mean_ratio=ratio,relative_change_pct=100*(ratio-1)))
    for records,file in [(means,"means.csv"),(differences,"paired_differences.csv")]:
        pd.DataFrame(records).to_csv(HERE/file,index=False,encoding="utf-8-sig")


if __name__=="__main__":
    main()
