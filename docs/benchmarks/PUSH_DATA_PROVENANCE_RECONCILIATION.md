# Push MRT Data Provenance Reconciliation

Date: 2026-11-16  
Branch: `codex/batch2-push-provenance`  
Evidence class: provenance audit only; no participant rows read or uploaded.

## Gate status

**BLOCKED / NOT_AVAILABLE for real local myBPmyLife execution.** The local raw archive is absent from the bounded machine audit, and the upstream repository exposes no machine-readable license metadata. No real Push result is produced.

## Local availability audit

The read-only filename audit searched the local Downloads and Projects roots for `Data_MRT.zip`, myBPmyLife archive names and MRT archive names. The only matches were repository code/documentation and a Python bytecode cache for the audit script; no raw `Data_MRT.zip` or participant-level archive was found.

Existing project evidence is reconciled as follows:

| Evidence source | Recorded state | Interpretation |
|---|---|---|
| Push P0/P1 reports | PASS for a supplied local snapshot | Historical supplied-snapshot evidence; archive fingerprint and license are not present in this worktree |
| Experiment A data audit | `NOT_AVAILABLE`, zero formal rows | Current bounded local evidence; no real MRT execution claimed |
| Project status | PARTIAL / NOT VERIFIED | Correct combined status until the exact archive/run is reconciled |

These records are not silently combined into a single real-data PASS.

## Upstream metadata audit

Source repository: [WIRED-L-um/MRT_myBPmyLife](https://github.com/WIRED-L-um/MRT_myBPmyLife)

- Repository is public and not archived.
- Main commit observed during this audit: `fa86c0d44ced3876f4a7351bc2d475fc4249342e`.
- Repository contents include `Data_MRT.zip` with remote GitHub contents size `13,389,920` bytes and Git blob SHA-1 `69eef5635d05974527ff3b306ac01aeaf4db7b34`.
- GitHub `licenseInfo` is null and the root listing contains no license file. This is not evidence that use or redistribution is permitted.
- The README documents the study design and proximal outcome definitions but does not establish a data license or a local archive fingerprint.

The remote Git blob SHA is a source locator, not a SHA-256 fingerprint of a locally obtained archive. The archive was not downloaded because the license/terms are unverified.

## Required missing evidence

Before a real Push lane can be marked available, obtain through an authorized channel and record outside raw Git content:

1. lawful source and license/terms approval;
2. local archive filename, byte size and SHA-256;
3. dictionary version and exact field mapping;
4. run ID, code commit, config and seed;
5. confirmation that the P0/P1 snapshot and Experiment A archive are identical;
6. participant, decision-point, timestamp/availability, assignment, delivery/exposure and outcome-window checks.

## Decision

Do not download, redistribute, or run real Push causal experiments until the missing license and archive evidence are supplied. Continue only with already authorized semi-synthetic fixtures and keep the real-data gate `BLOCKED/NOT_AVAILABLE`.
