"""Rename the Arena agent or refresh its public profile via ``PATCH /agent/me``.

Credentials resolve through :mod:`tools.api` (``ARENA_CREDENTIALS`` when set,
otherwise the repository's gitignored ``.arena-credentials``); the key is never
printed. Dry-run by default -- it shows the current profile and the intended
body, and only ``--apply`` sends the request.

Owner rule (2026-08-13): the public ``quote`` and ``description`` stay exactly
``Hello`` across every version and rename, so opponents can read nothing about
the policy. ``--quote``/``--description`` exist for a deliberate owner override
and are not part of routine version updates.

The display ``name`` and the public ``handle`` are separate fields and the
handle does **not** follow a rename: the agent became ``0Fold`` on 2026-08-26
while its handle stayed ``fold_ver_3``.

**The handle cannot be changed here.** Measured 2026-08-26: sending it returns
``HTTP 400 {"error":"Error","message":"body must not have additional
properties"}``. The endpoint validates against a closed schema of
``name``/``quote``/``description``, so ``handle`` is rejected rather than
ignored, and the rejection discards the *whole* body -- a PATCH carrying a
handle changes nothing at all. ``--handle`` is kept so the failure is
reproducible rather than rediscovered, and the per-field verification after
the PATCH reports which fields actually moved (an API that ignored an unknown
key would answer 200 having done nothing).

Usage::

    python tools/update_agent_profile.py                      # show profile
    python tools/update_agent_profile.py --name Fold-ver-5 --apply
    python tools/update_agent_profile.py --handle 0fold --apply
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api import BASE, load_key  # noqa: E402  (sibling tool module)

PUBLIC_TEXT = "Hello"
SECRET_HINTS = ("key", "token", "secret", "credential")


def request(
    key: str, method: str, path: str, body: dict[str, Any] | None = None
) -> tuple[int, Any]:
    """Issue one Arena request; Arena paths only, so the key cannot leak."""

    if not path.startswith("/") or path.startswith("//"):
        raise ValueError("Arena API paths must start with one slash")
    payload = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"x-arena-api-key": key, "User-Agent": "curl/8.9.1"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    http_request = urllib.request.Request(
        BASE + path, data=payload, method=method, headers=headers
    )
    try:
        with urllib.request.urlopen(http_request, timeout=30) as response:
            raw, status = response.read().decode("utf-8"), response.status
    except urllib.error.HTTPError as error:
        raw, status = error.read().decode("utf-8", "replace"), error.code
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw[:2000]


def redact(value: Any) -> Any:
    """Drop key-like fields so a printed profile can never carry a secret."""

    if isinstance(value, dict):
        return {
            name: redact(item)
            for name, item in value.items()
            if not any(hint in name.lower() for hint in SECRET_HINTS)
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update the Arena agent profile.")
    parser.add_argument("--name", help="new display name, e.g. Fold-ver-5")
    parser.add_argument(
        "--handle",
        help=(
            "new public handle, e.g. 0fold. SEPARATE from --name and it "
            "does NOT follow it: the agent was renamed to '0Fold' on "
            "2026-08-26 while the handle stayed 'fold_ver_3'. Handles are "
            "public identifiers and may be immutable or unique-constrained, "
            "so the server may reject this or accept the request and ignore "
            "the field -- the per-field verification below is what tells "
            "you which"
        ),
    )
    parser.add_argument(
        "--quote", default=PUBLIC_TEXT, help=f"public tagline (default {PUBLIC_TEXT!r})"
    )
    parser.add_argument(
        "--description",
        default=PUBLIC_TEXT,
        help=f"public description (default {PUBLIC_TEXT!r})",
    )
    parser.add_argument("--apply", action="store_true", help="send the PATCH")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    key = load_key()

    status, profile = request(key, "GET", "/api/arena/agent/me")
    print(f"## GET /api/arena/agent/me -> HTTP {status}")
    print(json.dumps(redact(profile), indent=2)[:3000])
    if status != 200:
        print("cannot read the profile; nothing was changed")
        return 1

    body: dict[str, Any] = {"quote": args.quote, "description": args.description}
    if args.name:
        body["name"] = args.name
    if args.handle:
        body["handle"] = args.handle
    print(f"\n## intended PATCH body: {json.dumps(body)}")
    if not args.apply:
        print("dry run only; re-run with --apply to send")
        return 0

    status, result = request(key, "PATCH", "/api/arena/agent/me", body)
    print(f"\n## PATCH /api/arena/agent/me -> HTTP {status}")
    print(json.dumps(redact(result), indent=2)[:2000])
    if status not in (200, 204):
        return 1

    status, after = request(key, "GET", "/api/arena/agent/me")
    print(f"\n## verify GET -> HTTP {status}")
    print(json.dumps(redact(after), indent=2)[:3000])

    # HTTP 200 is not proof a field changed. An API that ignores unknown
    # keys answers 200 having done nothing, so compare what was asked for
    # against what came back, field by field.
    if isinstance(after, dict):
        print("\n## per-field verification")
        for field, wanted in body.items():
            actual = after.get(field)
            verdict = "APPLIED" if actual == wanted else "NOT APPLIED"
            print(f"  {field}: asked {wanted!r}, now {actual!r} -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
