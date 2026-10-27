# Disclosure-ask harness

Offline check for `.github/workflows/disclosure-ask.yml`, the QEP-5 disclosure ask
(QuantEcon/QuantEcon.py#965, QuantEcon/project-qep5#8). The workflow runs on
`pull_request_target`, so it cannot fire on the pull request that changes it; this
harness is the check a wording or filter change gets before merge.

It reads the inline script out of the workflow file and runs it against a fake GitHub
client, one case per skip rule and per way a disclosure can be present or absent.
Node only, no dependencies, not collected by `pytest`.

```bash
node ci/disclosure-ask/harness.mjs
```

The last lines print the comment as it would be posted. For the live checks after
merge (a `workflow_dispatch` with `dry_run`, then one real dispatch), see the pull
request that introduced the workflow.
