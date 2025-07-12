"""Helper wrappers around the **e2b** Python SDK.

We purposely keep this file tiny and dependency-free beyond `e2b` to simplify
unit testing (we can monkeypatch the SDK at import-time).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

# Classic e2b Client import removed – we only use DesktopSandbox for now

try:
    from e2b_desktop import Sandbox as DesktopSandbox  # type: ignore
except ImportError:  # pragma: no cover – desktop control optional
    DesktopSandbox = None  # type: ignore

# Global cache to keep DesktopSandbox instances alive across node invocations
_DESKTOP_CACHE: Dict[str, "DesktopSandbox"] = {}


__all__ = [
    "SandboxDetails",
    "create_or_get",
    "expires_in",
]


class SandboxDetails:  # noqa: D101 – simple data container
    def __init__(self, sandbox_id: str, url: str, expires_at: datetime):
        self.sandbox_id = sandbox_id
        self.url = url
        self.expires_at = expires_at

    # Allow truthiness checks – a deleted sandbox evaluates to False
    def __bool__(self) -> bool:  # noqa: D401 – special method
        return bool(self.sandbox_id and self.url)

    def as_dict(self) -> dict[str, str]:  # noqa: D401 – helper
        return {
            "sandbox_id": self.sandbox_id,
            "url": self.url,
            "expires_at": self.expires_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def create_or_get(
    *,
    api_key: str,
    template: str,
    timeout: int,
    sandbox_id: Optional[str] = None,
) -> SandboxDetails:
    """Return a **running** sandbox, creating it if necessary.

    Parameters
    ----------
    api_key
        e2b API key.
    template
        Name of the e2b template to launch when `sandbox_id` is *None*.
    timeout
        Desired sandbox lifetime in **seconds**.
    sandbox_id
        Optional ID of an existing sandbox to re-use.
    """

    client = Client(api_key=api_key)

    if sandbox_id:
        sandbox = client.sandboxes.get_sandboxes_sandbox_id(sandbox_id)
    else:
        sandbox = client.sandboxes.post_sandboxes(
            {
                "template": template,
                "timeout": timeout,
            }
        )
    return SandboxDetails(
        sandbox_id=sandbox["id"],  # type: ignore[index]
        url=sandbox["url"],  # type: ignore[index]
        expires_at=_parse_dt(sandbox["expiresAt"]),  # type: ignore[index]
    )


def expires_in(details: SandboxDetails) -> timedelta:
    """Return *timedelta* until sandbox expiry (negative if already expired)."""

    return details.expires_at - datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Desktop sandbox helper
# ---------------------------------------------------------------------------


def start_desktop_sandbox(*, timeout: int) -> SandboxDetails:
    """Spin up a **desktop** sandbox via `e2b_desktop.Sandbox`.

    Returns a `SandboxDetails` instance and stores the live DesktopSandbox object
    in an internal cache so that other nodes can reuse it.
    """

    if DesktopSandbox is None:
        raise RuntimeError(
            "e2b_desktop is not installed. Install it with `pip install e2b-desktop` to use desktop sandboxes."  # noqa: E501
        )

    desktop = DesktopSandbox()
    sandbox_id = desktop.sandbox_id  # type: ignore[attr-defined]

    # Start streaming immediately so planner / UI can display it
    desktop.stream.start(require_auth=True)  # type: ignore[attr-defined]
    auth_key = desktop.stream.get_auth_key()  # type: ignore[attr-defined]
    url = desktop.stream.get_url(auth_key=auth_key)  # type: ignore[attr-defined]

    expires_at = datetime.now(timezone.utc) + timedelta(seconds=timeout)

    _DESKTOP_CACHE[sandbox_id] = desktop  # type: ignore[arg-type]

    return SandboxDetails(sandbox_id=sandbox_id, url=url, expires_at=expires_at)


def get_desktop(sandbox_id: str):  # noqa: D401 – helper
    """Return the cached DesktopSandbox for *sandbox_id* (or *None*)."""

    return _DESKTOP_CACHE.get(sandbox_id)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_dt(value: str) -> datetime:  # noqa: D401 – helper
    return datetime.fromisoformat(value.replace("Z", "+00:00")) 