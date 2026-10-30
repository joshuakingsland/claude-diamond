"""NOVIG-V3 request signing.

The canonical string is six lines joined by LF, no trailing newline:

    NOVIG-V3
    {unix_millis}
    {METHOD}
    {path}
    {canonical_query}
    {lowercase_hex(sha256(raw_body))}

The path is signed verbatim, never normalised or percent-decoded. The query
is decoded and re-encoded so that ``%2f`` and ``%2F`` sign the same, but a
bare ``+`` stays a literal ``+``. ``tests/novig_signing_vectors.json`` is the
exchange's own vector file and is the authority when this disagrees with
the prose docs.
"""

import base64
import hashlib

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.hazmat.primitives.serialization import load_pem_private_key

SCHEME = "NOVIG-V3"

_UNRESERVED = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
_HEX = frozenset(b"0123456789abcdefABCDEF")


def _decode(part):
    """Decode ``%XX`` escapes only. A malformed escape stays literal."""
    raw = part.encode("utf-8")
    out = bytearray()
    i = 0
    while i < len(raw):
        if (raw[i] == ord("%") and i + 2 < len(raw)
                and raw[i + 1] in _HEX and raw[i + 2] in _HEX):
            out.append(int(raw[i + 1:i + 3], 16))
            i += 3
        else:
            out.append(raw[i])
            i += 1
    return bytes(out)


def _encode(raw):
    return "".join(chr(b) if b in _UNRESERVED else f"%{b:02X}" for b in raw)


def canonical_query(raw_query):
    """Line 5 of the canonical string, built from the query as sent."""
    pairs = []
    for pair in raw_query.split("&"):
        if not pair:
            continue
        name, _, value = pair.partition("=")
        pairs.append((_encode(_decode(name)), _encode(_decode(value))))
    # Encoded parts are pure ASCII, so str ordering is bytewise ordering.
    pairs.sort()
    return "&".join(f"{name}={value}" for name, value in pairs)


def string_to_sign(timestamp_ms, method, path, raw_query, body):
    """The six-line canonical string. ``body`` is the exact bytes sent."""
    return "\n".join([
        SCHEME,
        str(int(timestamp_ms)),
        method.upper(),
        path,
        canonical_query(raw_query),
        hashlib.sha256(body).hexdigest(),
    ])


def load_private_key(pem_bytes):
    key = load_pem_private_key(pem_bytes, password=None)
    if not isinstance(key, (ed25519.Ed25519PrivateKey,
                            ec.EllipticCurvePrivateKey)):
        raise ValueError("Novig keys are Ed25519 or P-256")
    if (isinstance(key, ec.EllipticCurvePrivateKey)
            and not isinstance(key.curve, ec.SECP256R1)):
        raise ValueError("an ECDSA key must be on P-256")
    return key


def sign(private_key, text):
    """Standard padded base64 of the raw signature over ``text``.

    P-256 signatures are DER, which is what ``cryptography`` returns; a raw
    ``r||s`` signature never verifies.
    """
    data = text.encode("utf-8")
    if isinstance(private_key, ed25519.Ed25519PrivateKey):
        signature = private_key.sign(data)
    else:
        signature = private_key.sign(data, ec.ECDSA(hashes.SHA256()))
    return base64.b64encode(signature).decode("ascii")
