# Release candidate 0.1.0

This candidate is locally qualified on macOS and is not yet tagged, submitted, or published.

## Passing evidence

- The unit, smoke, typecheck, production build, production dependency audit, and reproducible packaging checks passed on 2026-09-02. Official Obsidian lint completed with no errors and 17 non-failing cross-window compatibility recommendations.
- The pinned Evaluated Configuration passed every quality, security, egress, and side-effect gate. See [`evaluation/results/evaluated-configuration.json`](evaluation/results/evaluated-configuration.json).
- The Reference Vault acceptance harness passed every gate once on macOS with Obsidian 1.13.7. See [`evaluation/results/acceptance.json`](evaluation/results/acceptance.json).
- The packaged `main.js` SHA-256 is `d1977a907af8793dce31a29ac945612119b6b227a4c97ef7a122e4fbd5f0e08c`.

## Pending gates

- Run the transport and acceptance probes on equivalent Windows and Linux hardware before claiming support for either platform.
- Resolve or explicitly review the remaining lint recommendations before submission.
- Run the Community Plugin automated review after an explicit submission action makes that service available.
- Create the exact `0.1.0` tag only after the pending gates pass against this bundle.
