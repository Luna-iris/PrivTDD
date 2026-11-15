"""Build the extended supplement without rewriting the frozen S1--S3 source."""
from pathlib import Path
import shutil
import subprocess
import pandas as pd


def _number(value, digits=5):
    value=f'{value:.{digits}g}'
    if 'e' in value:
        mantissa,power=value.split('e')
        return mantissa+r'\times 10^{'+str(int(power))+'}'
    return value


def build_supplement(root, rendered, output, compile_pdf=False):
    root,rendered,output=map(Path,(root,rendered,output))
    output.mkdir(parents=True,exist_ok=True)
    base=(root/'paper/supplement/published_e1_e3.tex').read_text(encoding='utf-8')
    if base.count(r'\end{document}')!=1:
        raise ValueError('Expected one document end in frozen supplement')
    data=pd.read_csv(rendered/'tables/e4_summary.csv',float_precision='round_trip')
    lines=[r'\clearpage',r'\section{Neighbor-pair sampling (E4)}',
      r'On Higgs social and Orkut, the total privacy budget is $\varepsilon=2$.',
      r'The four requested sampling rates are $0.01\%,0.1\%,1\%,10\%$.',
      r'Each rate uses seeds 20261028--20261032 and estimates the TDD of the full network.',
      r'The rate concerns candidate neighbor pairs after projection, not a node subsample.',
      r'The frontend retains its matching/rotation construction and corresponding sampled likelihood.',
      r'The metadata record the realized rate, design parameter, candidates, visits, and random streams.',
      r'Shared seed indices across rates do not imply identical private reports.',
      r'The $100\%$ row in the main paper comes from the ten E1 runs (seeds 20261028--20261037).',
      r'The five matched full-enumeration controls retained with E4 are separate auxiliary records.',
      r'The table below uses the main-paper control convention, and gives pointwise 95\% percentile',
      r'bootstrap intervals from 2,000 resamples with seed 731.',
      r'\begingroup\fontsize{8.3}{10}\selectfont\setlength{\tabcolsep}{3pt}',
      r'\begin{longtable}{llrrrr}',
      r'\caption{E4 errors in original units. The full-enumeration control uses ten runs; each sampled setting uses five.}\label{tab:e4:full}\\',
      r'\toprule Network & Rate & Runs & MSE & EMD & KS \\ \midrule\endfirsthead',
      r'\toprule Network & Rate & Runs & MSE & EMD & KS \\ \midrule\endhead',
      r'\bottomrule\endlastfoot']
    for ds in ['Higgs social','Orkut']:
        for rate in [.0001,.001,.01,.1,1.]:
            group=data[data.dataset.eq(ds)&data.rate.eq(rate)].set_index('metric')
            values=[]
            for metric in ['MSE','EMD','KS']:
                row=group.loc[metric]
                values.append(r'\compactCI{'+_number(row['mean'])+'}{'+_number(row['lower95'],4)+'}{'+_number(row['upper95'],4)+'}')
            lines.append(' & '.join([ds,f'{100*rate:g}\\%',str(int(group.iloc[0]['n'])),*values])+r' \\')
    lines.extend([r'\end{longtable}',r'\endgroup'])
    e4='\n'.join(lines)+'\n'
    # Keep explanatory prose separate from auto-generated tables.
    (output/'e4.tex').write_text(e4,encoding='utf-8')
    shutil.copyfile(root/'paper/supplement/implementation.tex',output/'implementation.tex')
    extended=base.replace(r'\end{document}',r'\input{e4.tex}'+'\n'+r'\input{implementation.tex}'+'\n'+r'\end{document}')
    extended=extended.replace(r'(p.~\pageref{sec:e3}).',r'(p.~\pageref{sec:e3}); E4 and implementation details follow.')
    (output/'supplementary.tex').write_text(extended,encoding='utf-8')
    if compile_pdf:
        executable=shutil.which('latexmk')
        if not executable:raise FileNotFoundError('latexmk is required to compile the supplement')
        subprocess.run([executable,'-pdf','-interaction=nonstopmode','-halt-on-error','supplementary.tex'],cwd=output,check=True,capture_output=True,text=True)
    return output/'supplementary.tex'
