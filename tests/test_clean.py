import sys
import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from clean import clean_frame, sha256, COLUMNS

class CleaningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / 'data/raw/H2.csv'
        cls.before = sha256(cls.path)
        cls.raw = pd.read_csv(cls.path, dtype='string', keep_default_na=False)
        cls.cleaned, cls.audit = clean_frame(cls.raw)

    def test_real_data_and_source(self):
        self.assertEqual(self.raw.shape, (79330, 31))
        self.assertEqual(self.cleaned.shape, (79330, 32))
        self.assertEqual(sha256(self.path), self.before)
        self.assertEqual(self.cleaned['Children'].isna().sum(), 4)
        self.assertEqual(self.cleaned['Country'].isna().sum(), 24)
        self.assertEqual((self.cleaned.Agent == 'NotApplicable').sum(), 8131)
        self.assertEqual((self.cleaned.Company == 'NotApplicable').sum(), 75641)
        self.assertEqual(str(self.cleaned.Children.dtype), 'Int64')
        self.assertEqual(str(self.cleaned.Agent.dtype), 'string')
        self.assertEqual(self.cleaned.ArrivalDate.iloc[0], '2015-07-01')
        self.assertEqual(self.audit['raw_exact_duplicate_rows'], 25902)
        self.assertEqual(self.audit['post_normalization_duplicate_rows'], 25902)

    def test_idempotent(self):
        again, _ = clean_frame(self.cleaned)
        pd.testing.assert_frame_equal(self.cleaned, again)

    def test_padded_markers_and_duplicate_retention(self):
        rows = pd.concat([self.raw.iloc[:1], self.raw.iloc[:1]], ignore_index=True)
        rows.loc[0, 'Children'] = ' NA '
        rows.loc[0, 'Agent'] = ' NULL '
        rows.loc[0, 'Country'] = '   '
        cleaned, _ = clean_frame(rows)
        self.assertTrue(pd.isna(cleaned.loc[0, 'Children']))
        self.assertTrue(pd.isna(cleaned.loc[0, 'Country']))
        self.assertEqual(cleaned.loc[0, 'Agent'], 'NotApplicable')
        dupes, audit = clean_frame(pd.concat([self.raw.iloc[:1]] * 2, ignore_index=True))
        self.assertEqual(len(dupes), 2)
        self.assertEqual(audit['raw_exact_duplicate_rows'], 1)

    def test_malformed_inputs_fail(self):
        sample = self.raw.iloc[:1].copy()
        with self.assertRaisesRegex(ValueError, 'schema'):
            clean_frame(sample.drop(columns=['ADR']))
        for column, value, message in [('ADR', 'oops', 'numeric'), ('Adults', '1.5', 'numeric'), ('ArrivalDateMonth', 'NotAMonth', 'date'), ('ArrivalDateDayOfMonth', '32', 'date'), ('ReservationStatusDate', 'invalid', 'date')]:
            broken = sample.copy()
            broken.loc[broken.index[0], column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, message):
                clean_frame(broken)

if __name__ == '__main__':
    unittest.main()
