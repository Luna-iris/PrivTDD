"""Reject Chinese text in the public artifact, including decoded JSON strings.

PDF text and figure labels are separately checked at release time. Scientific
symbols and accented Latin names are permitted; this is not an ASCII-only rule.
"""
import argparse
import json
from pathlib import Path
import re

HAN=re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]')
ROOTS={'.github','src','tests','configs','docs','environment','reproduction','provenance','paper','LICENSES','datasets','results'}
TOP={'.gitignore','.gitattributes','CITATION.cff','LICENSE','pyproject.toml','README.md','THIRD_PARTY.md'}
SKIP={'.git','.venv','__pycache__','.pytest_cache','build','dist','work'}
LOCAL=('datasets/raw','datasets/processed','datasets/truth','results/runs','results/cache','results/archive','results/reproduced','results/checks','results/validation')
BINARY={'.pdf','.png','.jpg','.jpeg','.gif','.npy','.npz','.zip','.whl','.pyc','.nbc','.nbi'}
TEMP={'.log','.aux','.bbl','.blg','.out','.fls','.fdb_latexmk'}


def check(root):
    inspected=0;issues=[]
    for path in sorted(root.rglob('*')):
        relative=path.relative_to(root)
        if not path.is_file() or any(p in SKIP or p.endswith('.egg-info') for p in relative.parts):continue
        if relative.parts[0] not in ROOTS and relative.as_posix() not in TOP:continue
        if any(relative.is_relative_to(Path(prefix)) for prefix in LOCAL):continue
        if path.suffix in TEMP or path.name.endswith('.synctex.gz'):continue
        if HAN.search(relative.as_posix()):issues.append({'path':relative.as_posix(),'kind':'filename'})
        if path.suffix.lower() in BINARY:continue
        text=path.read_text(encoding='utf-8-sig');inspected+=1
        if path.suffix=='.json':text=json.dumps(json.loads(text),ensure_ascii=False,indent=2)
        for number,line in enumerate(text.splitlines(),1):
            if HAN.search(line):issues.append({'path':relative.as_posix(),'line':number,'kind':'Chinese text'})
    return {'status':'passed' if not issues else 'failed','text_files_checked':inspected,'issues':issues,
            'binary_scope':'PDF extracted text and visible figure labels are audited separately'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();result=check(args.root.resolve())
    print(json.dumps(result,indent=2));return int(bool(result['issues']))


if __name__=='__main__':raise SystemExit(main())
