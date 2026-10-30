import base64
import json
import unittest
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from novig.client import encode_query
from novig.signing import (canonical_query, load_private_key, sign,
                           string_to_sign)

VECTORS = json.loads(
    (Path(__file__).parent / "novig_signing_vectors.json").read_text())


class VectorTests(unittest.TestCase):
    """The exchange's own 30 vectors; any diff here fails every request."""

    def test_canonical_strings_match_every_vector(self):
        for vector in VECTORS["vectors"]:
            given = vector["input"]
            with self.subTest(vector["id"]):
                self.assertEqual(
                    string_to_sign(given["timestamp"], given["method"],
                                   given["path"], given["query"],
                                   given["body"].encode("utf-8")),
                    vector["string_to_sign"])

    def test_signatures_verify_against_published_public_keys(self):
        for vector in VECTORS["vectors"]:
            pair = VECTORS["keypairs"][vector["keypair_id"]]
            private = load_private_key(
                pair["private_key_pkcs8_pem"].encode())
            public = load_pem_public_key(pair["public_key_spki_pem"].encode())
            text = vector["string_to_sign"]
            raw = base64.b64decode(sign(private, text))
            with self.subTest(vector["id"]):
                if vector["algorithm"] == "ed25519":
                    self.assertEqual(len(raw), 64)
                    public.verify(raw, text.encode())
                else:
                    public.verify(raw, text.encode(),
                                  ec.ECDSA(hashes.SHA256()))
                if vector.get("signature") and vector["algorithm"] == "ed25519":
                    # Ed25519 is deterministic, so a published signature
                    # must be reproduced byte for byte.
                    self.assertEqual(sign(private, text), vector["signature"])


class QueryTests(unittest.TestCase):
    def test_sent_query_is_already_canonical(self):
        query = encode_query({"league": "ATP,WTA", "limit": 5000,
                              "eventStatus": "OPEN_INGAME",
                              "marketType": None})
        self.assertEqual(query, canonical_query(query))
        self.assertNotIn("marketType", query)
        self.assertIn("league=ATP%2CWTA", query)

    def test_malformed_escape_stays_literal(self):
        self.assertEqual(canonical_query("q=100%"), "q=100%25")


if __name__ == "__main__":
    unittest.main()
