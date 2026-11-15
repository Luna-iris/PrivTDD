# Prepared datasets on Google Drive

Google Drive can host the large prepared inputs while GitHub contains the code,
small evidence tables, download instructions, and checksums. The local release
contains one ZIP per dataset in the ignored `bundles/` directory. Each contains
CSR arrays, exact triangle truth, the node-ID map, and portable graph metadata.
No third-party code, historical machine paths, or working caches are included.

`datasets/cloud_manifest.json` records the exact ZIP and member hashes, public
file links and validation status. The maintainer's
[Drive folder](https://drive.google.com/drive/folders/1xuxagBfpbpQYqq7mOQy0kVtiY15FG-zW?usp=sharing)
contains the prepared bundles. Keep the original dataset citations and terms
in `datasets/manifest.json`; a Drive mirror does not change those terms or
establish redistribution permission.

## Publishing a mirror

For each dataset whose distribution conditions are resolved, upload its ZIP
as a binary file without conversion. Configure downloadable Viewer access and
test the file link without an authenticated account. Record that **file** link
in the corresponding `share_url`, together with the release version. Folder
links alone are not sufficient for the downloader. Do not replace a released
ZIP with different bytes under the same release identity.

Google documents [link sharing and Viewer access](https://support.google.com/drive/answer/2494822).
Public links may face [temporary download-volume restrictions](https://support.google.com/drive/answer/2423534).
The official-source preparation route therefore remains available. Drive is a
convenient mirror, while a later versioned archive can provide a persistent
reference for the paper. If anonymous review is needed, note that Google's
sharing documentation says a shared file exposes its owner's name and email.

## Download and install

Download one dataset from the recorded public file link:

```bash
python -m pip install gdown==5.2.0
python reproduction/data_bundles.py fetch --dataset higgs
python -m privtdd data verify --datasets higgs
```

To test an owner-provided link before it is recorded in the manifest:

```bash
python reproduction/data_bundles.py fetch --dataset higgs --share-url "https://drive.google.com/file/d/FILE_ID/view"
```

The optional [gdown downloader](https://github.com/wkentaro/gdown/tree/v5.2.0)
handles Drive's file-download flow. The tool uses public access, checks the
archive hash and every member hash, and only installs the documented dataset
paths. Existing differing data causes a clear failure rather than overwriting
an experiment input. The mirror's anonymous-download verification is recorded
in `provenance/drive_download_validation.json`. A file still being uploaded or
verified is explicitly marked; it is not counted as a successful validation.

For ZIPs downloaded manually into `bundles/`:

```bash
python reproduction/data_bundles.py verify
python reproduction/data_bundles.py install
```

For original-source preparation without a Drive mirror:

```bash
python -m privtdd data prepare
```

Maintainers can regenerate local bundle candidates from verified prepared data:

```bash
python reproduction/data_bundles.py build
```

Rebuilding may produce a new container checksum with a different compression
implementation. Verify member identities and create a new manifest/release
instead of silently changing an already published checksum.
