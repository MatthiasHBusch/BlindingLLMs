"""
Headless-friendly Polaris Hub login.

The normal `polaris login` reads the OAuth authorization code via input(), which
fails in this background/non-TTY harness ("RuntimeError: lost sys.stdin"). This
helper performs the identical OAuth2 PKCE flow but reads the code from a FILE,
so the whole exchange happens inside one persistent process (the code_verifier
must match between URL generation and token exchange -> same process).

Usage:
    python polaris_login_helper.py <code_file>
It prints `AUTHORIZE_URL: <url>`, then waits for <code_file> to be created with
the authorization token in it, exchanges it, and caches the Hub credentials.
"""

import os
import sys
import time

from polaris.hub.client import PolarisHubClient

CODE_FILE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/polaris_code.txt"
TIMEOUT_S = 600

# Start clean so a stale file from a previous attempt is never reused.
try:
    os.remove(CODE_FILE)
except FileNotFoundError:
    pass

with PolarisHubClient() as client:
    ext = client.external_client

    # Step 1: build the authorization URL (binds this process's code_verifier).
    authorization_url, _state = ext.create_authorization_url()
    print(f"AUTHORIZE_URL: {authorization_url}", flush=True)
    print("Waiting for authorization code file...", flush=True)

    # Step 2: wait for the code to be written to CODE_FILE.
    deadline = time.time() + TIMEOUT_S
    while not os.path.exists(CODE_FILE):
        if time.time() > deadline:
            print("LOGIN_TIMEOUT", flush=True)
            sys.exit(2)
        time.sleep(1)
    with open(CODE_FILE) as f:
        code = f.read().strip()
    print("Got authorization code, exchanging for token...", flush=True)

    # Step 3: external OAuth code -> external token (cached on disk).
    ext.fetch_token(code=code, grant_type="authorization_code")

    # Step 4: external token -> Polaris Hub token (cached on disk).
    client.token = client.fetch_token()

    email = None
    try:
        email = ext.user_info.get("email")
    except Exception:
        pass
    print(f"LOGIN_OK {email}", flush=True)
