"""Build, fetch, verify and install versioned prepared-data bundles.

Bundles are local files until an owner publishes them and records their URLs.
The repository stores the manifest, not the large archives. This tool never
uploads files or changes cloud sharing permissions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from urllib.parse import parse_qs, urlparse
import zipfile


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(root, output, manifest_path):
    registry=json.loads((root/'datasets/manifest.json').read_bytes())
    output.mkdir(parents=True,exist_ok=True)
    result={'schema_version':1,'release':'paper-v1','provider':'Google Drive (optional mirror)',
            'status':'prepared locally; not uploaded','bundles':{}}
    for dataset,entry in registry['datasets'].items():
        source=root/'datasets/processed'/dataset
        names=sorted(set(entry['processed_files'])|{'graph.json'})
        members=[]
        for name in names:
            path=source/name
            checksum=digest(path)
            expected=entry['processed_files'].get(name)
            if expected and (checksum!=expected['sha256'] or path.stat().st_size!=expected['bytes']):
                raise ValueError(f'Prepared input does not match the paper registry: {dataset}/{name}')
            members.append({'path':f'datasets/processed/{dataset}/{name}',
                            'sha256':checksum,'bytes':path.stat().st_size})
        filename=f'privtdd-paper-v1-{dataset}-prepared.zip'
        archive=output/filename
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as bundle:
            for member in members:
                info=zipfile.ZipInfo(member['path'],date_time=(2026,1,1,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED
                info.external_attr=0o100644<<16
                with (root/member['path']).open('rb') as source_stream,bundle.open(info,'w',force_zip64=True) as target:
                    shutil.copyfileobj(source_stream,target,1024*1024)
        result['bundles'][dataset]={'filename':filename,'bytes':archive.stat().st_size,'sha256':digest(archive),
             'share_url':None,'source_page':entry['source_page'],'source_terms':entry['license_status'],
             'content':'prepared CSR, node-ID map, exact triangle truth, and portable graph metadata',
             'files':members}
        print(f'Built {filename}: {archive.stat().st_size} bytes',flush=True)
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    manifest_path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def check_archive(archive,entry):
    if archive.stat().st_size!=entry['bytes'] or digest(archive)!=entry['sha256']:
        raise ValueError(f'Archive checksum or size mismatch: {archive}')
    expected={item['path']:item for item in entry['files']}
    if len(expected)!=len(entry['files']):raise ValueError('Duplicate manifest paths')
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.namelist())!=len(expected) or set(bundle.namelist())!=set(expected):
            raise ValueError('Archive members do not match the manifest')
        for name,item in expected.items():
            relative=PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
                raise ValueError(f'Invalid bundle path: {name}')
            if relative.parts[:2]!=('datasets','processed'):raise ValueError(f'Unexpected bundle path: {name}')
            info=bundle.getinfo(name)
            if info.file_size!=item['bytes']:raise ValueError(f'Wrong member size: {name}')
            with bundle.open(name) as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=item['sha256']:
                    raise ValueError(f'Wrong member checksum: {name}')


def install(root,archive,entry):
    check_archive(archive,entry)
    for item in entry['files']:
        target=(root/item['path']).resolve()
        if not target.is_relative_to(root.resolve()):raise ValueError('Destination escapes artifact root')
        if target.exists() and digest(target)!=item['sha256']:
            raise FileExistsError(f'Existing data differs; select a separate artifact root: {target}')
    with zipfile.ZipFile(archive) as bundle:
        for item in entry['files']:
            target=root/item['path']
            if target.exists():continue
            target.parent.mkdir(parents=True,exist_ok=True)
            partial=target.with_name(target.name+'.bundle-part')
            with bundle.open(item['path']) as source,partial.open('wb') as destination:
                shutil.copyfileobj(source,destination,1024*1024)
            if digest(partial)!=item['sha256']:raise ValueError(f'Installed checksum mismatch: {target}')
            partial.replace(target)


def google_drive_url(value):
    url=urlparse(value)
    if url.scheme!='https' or url.hostname!='drive.google.com':
        raise ValueError('Use an HTTPS drive.google.com file share link')
    identifier=parse_qs(url.query).get('id',[None])[0]
    parts=url.path.split('/')
    if 'd' in parts and len(parts)>parts.index('d')+1:identifier=parts[parts.index('d')+1]
    if not identifier or not all(c.isalnum() or c in '_-' for c in identifier):
        raise ValueError('A Google Drive file link is required; folder links are not supported')
    return 'https://drive.google.com/uc?id='+identifier


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['build','verify','install','fetch'])
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--bundle-dir',type=Path)
    parser.add_argument('--dataset',choices=['bio-dmela','dblp','higgs','orkut'])
    parser.add_argument('--share-url',help='Explicit Drive file URL when the manifest URL is not yet set')
    args=parser.parse_args(argv)
    root=args.repo_root.resolve()
    manifest_path=args.manifest or root/'datasets/cloud_manifest.json'
    output=args.bundle_dir or root/'bundles'
    if args.action=='build':build(root,output,manifest_path);return 0
    manifest=json.loads(manifest_path.read_bytes())
    if args.action=='fetch' and not args.dataset:parser.error('fetch requires --dataset')
    selected=[args.dataset] if args.dataset else list(manifest['bundles'])
    for dataset in selected:
        entry=manifest['bundles'][dataset];archive=output/entry['filename']
        if args.action=='fetch' and not archive.exists():
            share_url=args.share_url or entry.get('share_url')
            if not share_url:raise ValueError('No published mirror URL is recorded. Use official data preparation or supply --share-url.')
            url=google_drive_url(share_url)
            try:import gdown
            except ImportError as error:raise RuntimeError('Install the optional downloader with: python -m pip install gdown==5.2.0') from error
            output.mkdir(parents=True,exist_ok=True)
            partial=archive.with_name(archive.name+'.download-part')
            downloaded=gdown.download(url=url,output=str(partial),quiet=False,use_cookies=False,resume=True)
            if not downloaded:raise RuntimeError('Drive download failed; check public Viewer access and download availability')
            check_archive(partial,entry)
            partial.replace(archive)
        if args.action in {'install','fetch'}:install(root,archive,entry)
        else:check_archive(archive,entry)
        print(f'{dataset}: {args.action} verified',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
