"""Tear down all `pyine-transferability-*` endpoints (Runpod only).

Run via: `make teardown` -- or directly:
    uv run --project $PYINE_ROOT python deploy/teardown_endpoints.py [--dry-run]

Deletes ENDPOINTS only. Templates are preserved across teardown/redeploy
cycles because:
  - They're inert (no compute, no cost) and don't expose persistent state.
  - `deploy_endpoints.py` reuses any existing template with the target name
    instead of creating a new one. Preserved templates make the next deploy
    faster and avoid the "Template name must be unique" collision that
    blocks re-creation.
  - They serve as a record of what config was deployed (image version, env
    vars, disk size). Deleting loses that record. The current template
    spec is documented in `deploy/README.md`.

Endpoints AND templates are looked up by NAME (prefix `pyine-transferability-`)
rather than by IDs in .env, so a stale .env never leaves orphans behind.
Templates are listed for visibility (especially useful to confirm the deploy
script's reuse path will find them on the next run) but never deleted by
this script.

GraphQL-based: the pinned runpod SDK (`runpod.api.mutations.endpoints` /
`runpod.api.mutations.templates`) only exposes `saveEndpoint`,
`updateEndpointTemplate`, and `saveTemplate`. There's no `delete_endpoint`
Python helper, so we hit the GraphQL endpoint directly via
`runpod.api.graphql.run_graphql_query`. Listing endpoints uses the SDK's
`get_endpoints()` (which works) and listing templates uses GraphQL (since
the SDK has no list-templates helper either).

WARNING: this deletes ALL Runpod endpoints with name prefix
`pyine-transferability-`. If you have other endpoints with similar names,
either rename them or change `NAME_PREFIX` below.

If you ever genuinely need to remove a template (e.g., to change its image
version), do it manually via the Runpod console -- there's no automation
here on purpose, because the default policy is to keep templates.

CLI:
    python teardown_endpoints.py            # list matching infra, then delete endpoints
    python teardown_endpoints.py --dry-run  # list only, no destructive ops
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import dotenv
import runpod
import runpod.api.graphql

_TRANSF_ROOT = pathlib.Path(__file__).resolve().parents[1]  # transferability/ (study root)
# study's own .env; override via TRANSF_DOTENV=...; legacy PYINE_DOTENV also accepted
ENV_PATH = pathlib.Path(os.environ.get("TRANSF_DOTENV", os.environ.get("PYINE_DOTENV", _TRANSF_ROOT / ".env")))
NAME_PREFIX = "pyine-transferability-"


def _list_endpoints_matching(prefix: str) -> list[dict]:
    """Return endpoint dicts whose `name` starts with `prefix`.

    Prefers the SDK's `get_endpoints()` because it includes the attached
    `template` reference. Falls back to a GraphQL query if the SDK helper is
    missing (older or stripped-down SDK builds).
    """
    try:
        endpoints = runpod.get_endpoints()
    except AttributeError:
        # SDK feature detection: fall back to GraphQL when the helper isn't exposed in the pinned SDK build.
        resp = runpod.api.graphql.run_graphql_query("query { myself { endpoints { id name template { id } } } }")
        endpoints = resp.get("data", {}).get("myself", {}).get("endpoints", []) or []
    return [endpoint for endpoint in endpoints if str(endpoint.get("name", "")).startswith(prefix)]


def _list_templates_matching(prefix: str) -> list[dict]:
    """Return template dicts (id, name) whose `name` starts with `prefix`.

    Always uses GraphQL: the SDK has no list-templates helper. Necessary for
    finding orphan templates that aren't currently attached to any endpoint.
    """
    resp = runpod.api.graphql.run_graphql_query("query { myself { podTemplates { id name } } }")
    templates = resp.get("data", {}).get("myself", {}).get("podTemplates", []) or []
    return [template for template in templates if str(template.get("name", "")).startswith(prefix)]


def _delete_endpoint(ep_id: str) -> str | None:
    """Delete an endpoint by ID via the `deleteEndpoint` GraphQL mutation.

    Returns None on success, an error-summary string on failure.
    """
    mutation = f'mutation {{ deleteEndpoint(id: "{ep_id}") }}'
    try:
        resp = runpod.api.graphql.run_graphql_query(mutation)
    except Exception as err:  # noqa: BLE001 -- per-endpoint teardown must surface error and continue with others
        return f"{type(err).__name__}: {err}"
    if "errors" in resp:
        return f"graphql: {resp['errors']}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching endpoints and templates without deleting anything.",
    )
    args = parser.parse_args()

    if not ENV_PATH.is_file():
        print(f"ERROR: env file not found at {ENV_PATH}", file=sys.stderr)
        return 1
    dotenv.load_dotenv(ENV_PATH)

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY missing", file=sys.stderr)
        return 1
    runpod.api_key = api_key

    endpoints = _list_endpoints_matching(NAME_PREFIX)
    templates = _list_templates_matching(NAME_PREFIX)

    print(f"# Found {len(endpoints)} endpoint(s) with prefix {NAME_PREFIX!r} (will be deleted):")
    for endpoint in endpoints:
        tpl_ref = (endpoint.get("template") or {}).get("id")
        print(f"  id={endpoint.get('id')}  name={endpoint.get('name')}  attached_template={tpl_ref}")
    print(f"# Found {len(templates)} template(s) with prefix {NAME_PREFIX!r} (will be PRESERVED):")
    for template in templates:
        print(f"  id={template.get('id')}  name={template.get('name')}")
    print(
        "# Templates persist across teardown/redeploy cycles by design "
        "(see module docstring + deploy/README.md). The next `make deploy` will reuse them."
    )
    print()

    if args.dry_run:
        print("# --dry-run: skipping endpoint deletion. Nothing was modified.")
        return 0

    if not endpoints:
        print("No endpoints to delete.")
        return 0

    failed = 0
    for endpoint in endpoints:
        ep_id = endpoint.get("id")
        ep_name = endpoint.get("name", "<unknown>")
        print(f"Deleting endpoint id={ep_id} name={ep_name} ...", flush=True)
        err = _delete_endpoint(ep_id)
        if err is None:
            print("  ok", flush=True)
        else:
            print(f"  FAILED: {err}", flush=True)
            failed += 1

    print()
    if failed == 0:
        print(
            f"Teardown complete. Deleted {len(endpoints)} endpoint(s); preserved "
            f"{len(templates)} template(s). Verify in https://console.runpod.io/serverless."
        )
        return 0
    print(
        f"Teardown finished with {failed} failure(s). Manual cleanup via "
        f"https://console.runpod.io/serverless may be required.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
