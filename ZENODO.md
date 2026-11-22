# Zenodo release procedure

Purpose: turn the frozen repository into a citable object with a DOI, so that the
Code Availability paragraph of the manuscript can name something permanent instead
of a GitHub URL that could change.

**The order matters twice over.** The content must be frozen before the DOI exists,
and — the step people miss — the Zenodo switch must be on *before* the GitHub
Release is published. Zenodo archives releases that are created while the switch is
on; a release published before it is simply never seen, and the fix is to publish
another one.

## Status of this release

| Item | State |
|---|---|
| Repository public | done — https://github.com/lbzero/tiny-object-assignment-protocol |
| Tag `v1.0.0` pushed | done |
| Licence | done — AGPL-3.0, in `LICENSE` and in the three metadata files |
| GitHub Release | **not yet** (deliberately: see the ordering note) |
| Zenodo DOI | **not yet** — this is the last blocker |

## A. Turn the Zenodo switch on first

1. Open https://zenodo.org and sign in with **Sign in with GitHub** (top right).
   Use the same GitHub account that owns the repository (`lbzero`); the integration
   only sees repositories that account can administer.
2. Go to https://zenodo.org/account/settings/github/ . If GitHub is not linked yet,
   press **Connect** and authorise Zenodo to read your public repositories.
3. Find `lbzero/tiny-object-assignment-protocol` in the list and flip its switch to
   **ON**. It will be in the "not yet archived" state.

Nothing is published at this point. The switch only says "archive the next release".

## B. Publish the GitHub Release

1. Open https://github.com/lbzero/tiny-object-assignment-protocol/releases/new
2. **Choose a tag**: pick the existing `v1.0.0`. Do not create a new tag.
3. **Release title**: `v1.0.0`.
4. **Describe this release**: paste the `v1.0.0` section of `CHANGELOG.md`. Two or
   three sentences is enough; this text becomes part of the archival record.
5. Leave "Set as the latest release" ticked. Do **not** tick "pre-release".
6. Press **Publish release**.

Within about a minute, Zenodo receives a webhook and starts archiving.

## C. Collect the DOI

1. Reload https://zenodo.org/account/settings/github/ — the repository should now
   show a DOI badge instead of the switch.
2. Click through to the record. You will see two DOIs:
   - **Concept DOI** (`10.5281/zenodo.NNNNNNN`) — always resolves to the newest
     version. Use it on a project page or in a CV.
   - **Version DOI** (`10.5281/zenodo.NNNNNNN`) — resolves to v1.0.0 exactly.
   **The manuscript cites the version DOI**, because a reader must be able to get
   the code that produced the reported numbers, not a later state.
3. Send both strings back; the version DOI is the one that goes into the paper.

## D. Fill the three places (done by the assistant, not by hand)

| Where | Field | Value |
|---|---|---|
| `60-paper/latex/sections/08-discussion.tex` | `\pending{Zenodo DOI}` | version DOI |
| `CITATION.cff` | `identifiers[0].value` | version DOI |
| `.zenodo.json` | `related_identifiers` (add a `doi` entry) | version DOI |

After filling, `grep -c "PLACEHOLDER still in text" main.log` must drop from 3 to 2
(the remaining two are Funding and Acknowledgements, which are author facts).

## If the Zenodo integration is unavailable

The manual route also works and produces the same kind of DOI:

1. Build the payload: `bash tools/make_archive.sh --out dist --version v1.0.0`
   (it refuses to package `.png`, `.jpg`, `.pt`, so imagery and weights cannot leak).
2. Zenodo → **New upload** → upload `dist/ltnet-repro-v1.0.0.tar.gz` → the
   `.zenodo.json` in the repository is used as metadata if you paste it into the
   "Metadata" form, or fill the form by hand from it.
3. Reserve a DOI, then publish.
4. Note that the archived object is then the tarball, not the repository; the
   concept/version distinction is the same.

## What must never be published

- AI-TODv2 imagery, in whole or in part, including the crops behind the case figure.
  The benchmark's own terms govern redistribution; check them before releasing any
  figure-generating output.
- Prediction dumps and checkpoints, unless those terms allow it.
- Any file containing a personal path, an account name, or a token.
