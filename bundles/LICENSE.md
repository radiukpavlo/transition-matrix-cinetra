# Licence of the fitted bundles

The files in this folder, and the F9 weight files attached to the GitHub release, are objects fitted on the ACDC
development set in the frozen run `v6-replication-20261003-1241`. Because they derive from the ACDC dataset, they
are licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** licence
(SPDX `CC-BY-NC-SA-4.0`; legal code: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode), as the ACDC
terms require.

Attribution: P. Radiuk, O. Barmak, L. Klymenko and I. Krak, CINETRA (2026), fitted on the Automated Cardiac
Diagnosis Challenge (ACDC) dataset of O. Bernard et al., IEEE Transactions on Medical Imaging 37(11):2514–2525, 2018.

Some files are scikit-learn objects stored with `joblib` (Python pickle). Load them only after checking their
SHA-256 digests against `EXPORT_MANIFEST.json` at the repository root.
