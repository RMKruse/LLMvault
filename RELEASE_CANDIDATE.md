# Release candidate 0.1.0

This candidate is locally packaged on macOS and is not yet tagged, submitted, or published.

## Passing evidence

- The unit, smoke, typecheck, production build, production dependency audit, and reproducible packaging checks passed on 2026-09-05. Official Obsidian lint completed with no errors and 17 non-failing cross-window compatibility recommendations.
- The pinned Evaluated Configuration passed every quality, security, egress, and side-effect gate. See [`evaluation/results/evaluated-configuration.json`](evaluation/results/evaluated-configuration.json).
- The packaged `main.js` SHA-256 is `8f50ad690860382023e49d147f8e71c67a773d2fa8829ff5b70309c93fb51f18`.

## Pending gates

- Run the Reference Vault acceptance harness on macOS against this bundle; the prior report is bound to an older source and bundle hash.
- Run the transport and acceptance probes on equivalent Windows and Linux hardware before claiming support for either platform.
- Resolve or explicitly review the remaining lint recommendations before submission.
- Run the Community Plugin automated review after an explicit submission action makes that service available.
- Create the exact `0.1.0` tag only after the pending gates pass against this bundle.
