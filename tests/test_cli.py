from unittest import mock
from pathlib import Path
from privtdd import cli


def test_explicit_prepare_enables_official_download():
    with mock.patch.object(cli,'artifact_root',return_value=Path('.')), mock.patch('privtdd.data.prepare',return_value=Path('prepared')) as prepare:
        assert cli.main(['data','prepare','--datasets','bio-dmela'])==0
        assert prepare.call_args.kwargs['allow_download'] is True


def test_data_verification_failure_is_nonzero():
    with mock.patch.object(cli,'artifact_root',return_value=Path('.')), mock.patch('privtdd.data.verify',return_value={'ok':False,'issues':['hash mismatch']}):
        assert cli.main(['data','verify','--datasets','bio-dmela'])==1
