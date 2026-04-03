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

    def test_mapping_dementia_and_smc(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "labels_dx.csv"
            csv_path.write_text(
                "PTID,DX\n"
                "002_S_0413,Dementia\n"
                "003_S_0001,SMC\n",
                encoding="utf-8",
            )

            lookup = build_label_lookup(csv_path)
            self.assertEqual(lookup.subject_to_class.get("002_S_0413"), 2)
            self.assertEqual(lookup.subject_to_class.get("003_S_0001"), 0)

    def test_prefers_dx_bl_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "labels_dx_bl.csv"
            csv_path.write_text(
                "PTID,DX,DX_bl\n"
                "002_S_0413,Dementia,AD\n"
                "003_S_0001,MCI,CN\n",
                encoding="utf-8",
            )

            lookup = build_label_lookup(csv_path)
            self.assertEqual(lookup.subject_to_class.get("002_S_0413"), 2)
            self.assertEqual(lookup.subject_to_class.get("003_S_0001"), 0)

    def test_dxchange_numeric_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "labels_dxchange.csv"
            csv_path.write_text(
                "PTID,DXCHANGE\n"
                "002_S_0413,5\n"
                "003_S_0001,7\n"
                "004_S_0002,9\n",
                encoding="utf-8",
            )

            lookup = build_label_lookup(csv_path)
            self.assertEqual(lookup.subject_to_class.get("002_S_0413"), 2)
            self.assertEqual(lookup.subject_to_class.get("003_S_0001"), 0)
            self.assertEqual(lookup.subject_to_class.get("004_S_0002"), 1)


if __name__ == "__main__":
    unittest.main()
