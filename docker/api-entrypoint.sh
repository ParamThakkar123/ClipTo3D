#!/bin/sh
# The API's launcher, and the only reason this is not a plain CMD: whether
# uvicorn may believe X-Forwarded-For is a deployment question, and an image
# that answers it once answers it wrongly for every deployment without a proxy.
#
# uvicorn's ProxyHeadersMiddleware rewrites the peer address from
# X-Forwarded-For. Under `--forwarded-allow-ips '*'` it does that for every
# caller, including one connecting directly — so `request.client.host` becomes
# whatever the client typed. service.py's client_identity() is careful to
# ignore X-Forwarded-For unless CLIPTO3D_TRUST_PROXY says a trusted proxy is in
# front, but its fallback is request.client.host, which uvicorn has already
# replaced. The rate limiter and the per-address signup cap then bucket on a
# value the caller chooses, and neither limits anything: a fresh forged header
# per request is a fresh bucket, and in `public` mode that is unmetered key
# minting.
#
# So the flags travel with CLIPTO3D_TRUST_PROXY rather than being baked in.
# That variable already means "a proxy I trust sets these headers", which is
# exactly the precondition uvicorn's middleware needs.
set -eu

# Globbing off: --forwarded-allow-ips=* is a literal argument, and `$proxy`
# below is deliberately unquoted so it splits into separate words.
set -f

case "$(printf %s "${CLIPTO3D_TRUST_PROXY:-0}" | tr '[:upper:]' '[:lower:]')" in
    1 | true | yes)
        # Matches the truthiness service.py applies to the same variable, so
        # the two halves cannot disagree about what "trusted" means.
        #
        # '*' is right here and not lax: reaching this line means an operator
        # asserted a proxy terminates every connection, and in a container the
        # peer is that proxy's address on a bridge network — not something
        # worth enumerating. Set CLIPTO3D_FORWARDED_ALLOW_IPS to pin it anyway.
        proxy="--proxy-headers --forwarded-allow-ips=${CLIPTO3D_FORWARDED_ALLOW_IPS:-*}"
        ;;
    *)
        # Not merely omitted — uvicorn enables proxy headers by default, so
        # leaving the flag off would keep the bypass and only narrow it to
        # callers arriving from 127.0.0.1.
        proxy="--no-proxy-headers"
        ;;
esac

exec uvicorn service:app \
    --host "${CLIPTO3D_HOST:-0.0.0.0}" \
    --port "${CLIPTO3D_PORT:-8000}" \
    --timeout-keep-alive 75 \
    $proxy "$@"
