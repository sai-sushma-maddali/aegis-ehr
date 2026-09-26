"""Document text extraction for clinical attachment uploads."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from core.document_extract import (  # noqa: E402
    compose_prompt_with_document,
    extract_document_text,
)


class TestDocumentExtract(unittest.TestCase):
    def test_txt_and_compose_keep_poisoned_payload(self):
        path = ROOT / "tests" / "poisoned_docs" / "mccorkle_lab_addendum.txt"
        text = extract_document_text(path.name, path.read_bytes())
        self.assertIn("SYSTEM NOTE", text)
        prompt = compose_prompt_with_document(
            "Please summarize this uploaded follow-up pathology addendum for Michael Mccorkle (MRN L1004).",
            path.name,
            text,
        )
        self.assertIn("Uploaded document: mccorkle_lab_addendum.txt", prompt)
        self.assertIn("SYSTEM NOTE", prompt)

    def test_rejects_unsupported_type(self):
        with self.assertRaises(ValueError):
            extract_document_text("notes.csv", b"a,b,c\n")


if __name__ == "__main__":
    unittest.main()
