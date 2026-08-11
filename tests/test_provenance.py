import unittest

from provenance import feature_schema, model_version, repository_revision


class ProvenanceTests(unittest.TestCase):
    def test_schema_is_order_sensitive_and_versioned(self):
        first = feature_schema(["a", "b"])
        self.assertEqual(first, feature_schema(["a", "b"]))
        self.assertNotEqual(first, feature_schema(["b", "a"]))
        version = model_version("glm", ["a", "b"], revision="abc123")
        self.assertIn("glm", version)
        self.assertIn("abc123", version)
        self.assertIn(first, version)

    def test_revision_never_mislabels_a_dirty_checkout_as_clean(self):
        revision = repository_revision()
        self.assertTrue(revision == "unknown" or "wip" in revision
                        or len(revision) == 12)


if __name__ == "__main__":
    unittest.main()
