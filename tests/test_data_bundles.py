"""Bundle integrity and installation boundaries, without a cloud account."""
import hashlib
import importlib.util
from pathlib import Path
import zipfile
import pytest

SPEC=importlib.util.spec_from_file_location('artifact_data_bundles',Path(__file__).resolve().parents[1]/'reproduction/data_bundles.py')
B=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def bundle(tmp_path,name='datasets/processed/example/graph.json'):
    payload=b'{"dataset":"example"}\n'
    archive=tmp_path/'data.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr(name,payload)
    entry={'bytes':archive.stat().st_size,'sha256':B.digest(archive),
           'files':[{'path':name,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}]}
    return archive,entry


def test_install_is_verified_and_idempotent(tmp_path):
    archive,entry=bundle(tmp_path);root=tmp_path/'artifact'
    B.install(root,archive,entry)
    target=root/entry['files'][0]['path'];timestamp=target.stat().st_mtime_ns
    B.install(root,archive,entry)
    assert target.stat().st_mtime_ns==timestamp
    assert B.digest(target)==entry['files'][0]['sha256']


def test_corrupt_container_is_rejected_before_install(tmp_path):
    archive,entry=bundle(tmp_path);archive.write_bytes(archive.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='checksum or size'):B.install(tmp_path/'artifact',archive,entry)
    assert not (tmp_path/'artifact').exists()


def test_path_traversal_is_rejected(tmp_path):
    archive,entry=bundle(tmp_path,'datasets/processed/../../../outside.txt')
    with pytest.raises(ValueError,match='Invalid bundle path'):B.install(tmp_path/'artifact',archive,entry)
    assert not (tmp_path/'outside.txt').exists()


def test_existing_different_input_is_preserved(tmp_path):
    archive,entry=bundle(tmp_path);root=tmp_path/'artifact'
    target=root/entry['files'][0]['path'];target.parent.mkdir(parents=True);target.write_bytes(b'existing')
    with pytest.raises(FileExistsError,match='Existing data differs'):B.install(root,archive,entry)
    assert target.read_bytes()==b'existing'


def test_drive_file_links_and_folder_rejection():
    expected='https://drive.google.com/uc?id=example_123'
    assert B.google_drive_url('https://drive.google.com/file/d/example_123/view?usp=sharing')==expected
    assert B.google_drive_url('https://drive.google.com/open?id=example_123')==expected
    with pytest.raises(ValueError,match='file link'):B.google_drive_url('https://drive.google.com/drive/folders/example_123')
    with pytest.raises(ValueError,match='HTTPS drive.google.com'):B.google_drive_url('https://example.org/file/d/example_123')
