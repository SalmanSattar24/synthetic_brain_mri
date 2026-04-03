from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.common.labels import build_label_lookup, extract_subject_id_from_path, lookup_subject_class


class TestLabels(unittest.TestCase):
    def test_extract_subject_id(self) -> None:
        p = Path("C:/x/002_S_0413/something.nii.gz")
        self.assertEqual(extract_subject_id_from_path(p), "002_S_0413")

    def test_build_and_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "labels.csv"
            csv_path.write_text("subject_id,diagnosis\n002_S_0413,AD\n003_S_0001,CN\n", encoding="utf-8")

            lookup = build_label_lookup(csv_path)
            self.assertEqual(lookup_subject_class(Path("/tmp/002_S_0413/scan_preproc.nii.gz"), lookup), 2)
            self.assertEqual(lookup_subject_class(Path("/tmp/003_S_0001/scan_preproc.nii.gz"), lookup), 0)


if __name__ == "__main__":
    unittest.main()
