#!/usr/bin/env bash
# Litestream WAL round-trip smoke test for the Speaker Registry SQLite.
#
# Verifies that the replication path used to share the registry between the
# Mac dev box and Azure VM actually works end-to-end:
#   1. Create a SQLite registry with one row.
#   2. `litestream replicate` it to a "remote" (a local dir — see note below).
#   3. Delete the local file.
#   4. `litestream restore` from the remote into a fresh path.
#   5. Confirm the row came back.
#
# Why a local dir instead of Azurite?
#   `litestream`'s `abs` driver hardcodes `*.blob.core.windows.net` as the
#   endpoint and offers no override. We can't point it at Azurite without
#   forking, so the network/auth half of "Azure ABS specifically" stays a
#   manual smoke (deploy/README.md §3). What's worth automating here is the
#   replication semantics — WAL → snapshot → restore — and the `file` driver
#   exercises exactly that codepath in litestream itself.
#
# Requires: docker, sqlite3.
# Idempotent: each run uses a fresh tmpdir and tears its container down.

set -euo pipefail

WORKDIR="$(mktemp -d -t sv-litestream-smoke-XXXXXX)"
echo "[smoke] using workdir: $WORKDIR"

LITESTREAM_NAME="sv-litestream-smoke-replicate"
LITESTREAM_CFG="$WORKDIR/litestream.yml"
DB_PATH="$WORKDIR/speakers.db"
RESTORE_PATH="$WORKDIR/restored.db"

cleanup() {
  set +e
  docker rm -f "$LITESTREAM_NAME" >/dev/null 2>&1 || true
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "[smoke] missing required tool: $1" >&2; exit 1; }
}

require docker
require sqlite3

# Fail early with a clear message if the daemon isn't running, so users don't
# get a wall of "connect: no such file" mid-script.
if ! docker info >/dev/null 2>&1; then
  echo "[smoke] FAIL: docker daemon not reachable. Start Docker Desktop / OrbStack and retry." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
echo "[smoke] (1/4) creating local registry with Katie row"
sqlite3 "$DB_PATH" <<'SQL'
PRAGMA journal_mode=WAL;
CREATE TABLE speakers (id TEXT PRIMARY KEY, display_name TEXT, created_at REAL, updated_at REAL);
INSERT INTO speakers VALUES ('sp_smoke01', 'Katie', strftime('%s','now'), strftime('%s','now'));
SQL

# ---------------------------------------------------------------------------
echo "[smoke] (2/4) writing litestream config"
cat > "$LITESTREAM_CFG" <<EOF
dbs:
  - path: $DB_PATH
    replicas:
      - type: file
        path: $WORKDIR/replica
        sync-interval: 200ms
        retention: 24h
EOF

# ---------------------------------------------------------------------------
echo "[smoke] (3/4) replicating db (waiting for snapshot)"
docker run -d --rm --name "$LITESTREAM_NAME" \
  -v "$WORKDIR:$WORKDIR" \
  litestream/litestream:0.5.0 \
  replicate -config "$LITESTREAM_CFG" >/dev/null

# Wait for the first snapshot to land before knocking out the source file —
# otherwise `restore` won't find a generation to replay.
for _ in $(seq 1 50); do
  [ -d "$WORKDIR/replica/generations" ] && break
  sleep 0.2
done
if [ ! -d "$WORKDIR/replica/generations" ]; then
  echo "[smoke] FAIL: litestream never produced a snapshot" >&2
  docker logs "$LITESTREAM_NAME" >&2 || true
  exit 1
fi
sleep 1  # let any pending WAL frame replicate
docker stop "$LITESTREAM_NAME" >/dev/null

# ---------------------------------------------------------------------------
echo "[smoke] (4/4) restoring to a fresh path and verifying row"
rm -f "$DB_PATH" "$DB_PATH-wal" "$DB_PATH-shm"
docker run --rm \
  -v "$WORKDIR:$WORKDIR" \
  litestream/litestream:0.5.0 \
  restore -config "$LITESTREAM_CFG" -o "$RESTORE_PATH" "$DB_PATH"

NAME=$(sqlite3 "$RESTORE_PATH" "SELECT display_name FROM speakers WHERE id = 'sp_smoke01';")
if [ "$NAME" != "Katie" ]; then
  echo "[smoke] FAIL: expected display_name='Katie', got '$NAME'" >&2
  exit 1
fi

echo "OK: round-trip restored Katie"
