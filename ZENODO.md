# Zenodo release procedure

The order matters: the repository content must be frozen **before** the DOI is
created, and the DOI must exist **before** the manuscript is submitted. Until both
are true, the manuscript keeps visible placeholders in the Code Availability
paragraph, and those placeholders block submission.

## 1. Freeze the content

```bash
cd 60-paper/repro
python -m pytest -q                # if tests exist; otherwise skip
bash tools/make_archive.sh \
    --traces-a /root/runs/p1/diag \
    --traces-b /root/runs/p1/diagB \
    --out dist --version v1.0.0
```

`make_archive.sh` refuses to package any `.png`, `.jpg` or `.pt` file, so imagery
and weights cannot leak into the archive by accident. Check the printed SHA-256 and
keep `dist/SHA256SUMS` with the submission records.

## 2. Create the GitHub repository

```bash
git init
git add .
git commit -m "v1.0.0: diagnostic protocol and traces for the JSTARS submission"
git branch -M main
git remote add origin https://github.com/<account>/<repo>.git
git push -u origin main
```

Before pushing, confirm:

- [ ] No `.pt`, no imagery, no prediction dumps, no absolute paths to a private machine.
- [ ] `README.md` renders and every command in it has been run at least once.
- [ ] A licence file is present (see §5).
- [ ] `CITATION.cff` parses (`pip install cffconvert && cffconvert --validate`).

## 3. Tag and release

```bash
git tag -a v1.0.0 -m "Version archived with the JSTARS submission"
git push origin v1.0.0
```

Create a GitHub Release from the tag, attach `dist/ltnet-repro-v1.0.0.tar.gz` and
`dist/SHA256SUMS`, and use the CHANGELOG entry as the release notes.

## 4. Archive on Zenodo and obtain the DOI

1. In Zenodo, link the GitHub account and enable archiving for this repository, or
   upload the tarball as a new record with `.zenodo.json` as metadata.
2. Reserve a DOI, then publish. Record both the **concept DOI** (all versions) and
   the **version DOI** (v1.0.0). The manuscript cites the version DOI.
3. Replace, in this order:
   - `CITATION.cff` → `identifiers[0].value` and `.zenodo.json` → `related_identifiers`;
   - the manuscript's Code Availability paragraph: repository URL and DOI.
4. Re-run `tools/make_archive.sh` only if the payload changed; otherwise the tarball
   is the one already archived, and the hash in `SHA256SUMS` must still match.

## 5. Licence — decide before the record is created

The code imports `ultralytics`, which is distributed under AGPL-3.0. Two workable
options:

- **AGPL-3.0** for this package, matching the dependency. Simplest and safest.
- **MIT or BSD-3** for the original scripts, with a README note that users must
  comply with the AGPL when they install and use `ultralytics`. Defensible because
  the scripts are separate programs, but it requires the note to be explicit.

Zenodo requires a licence for software records, so this cannot be deferred past the
record creation. It is the one decision that is not mine to make.

## 6. What must never be published

- AI-TODv2 imagery, in whole or in part, including crops used for case figures.
  Case figures in the paper are derived from the benchmark; check its terms before
  releasing the figure-generating script's output.
- Prediction dumps and checkpoints, unless the benchmark terms allow it.
- Any file containing a personal path, an account name, or a token.
