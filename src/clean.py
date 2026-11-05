"""Conservative, deterministic cleaning for the H2 hotel booking dataset."""
from __future__ import annotations
import argparse
import calendar
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

COLUMNS = ['IsCanceled','LeadTime','ArrivalDateYear','ArrivalDateMonth','ArrivalDateWeekNumber','ArrivalDateDayOfMonth','StaysInWeekendNights','StaysInWeekNights','Adults','Children','Babies','Meal','Country','MarketSegment','DistributionChannel','IsRepeatedGuest','PreviousCancellations','PreviousBookingsNotCanceled','ReservedRoomType','AssignedRoomType','BookingChanges','DepositType','Agent','Company','DaysInWaitingList','CustomerType','ADR','RequiredCarParkingSpaces','TotalOfSpecialRequests','ReservationStatus','ReservationStatusDate']
INTEGER = ['IsCanceled','LeadTime','ArrivalDateYear','ArrivalDateWeekNumber','ArrivalDateDayOfMonth','StaysInWeekendNights','StaysInWeekNights','Adults','Children','Babies','IsRepeatedGuest','PreviousCancellations','PreviousBookingsNotCanceled','BookingChanges','DaysInWaitingList','RequiredCarParkingSpaces','TotalOfSpecialRequests']
CATEGORICAL = [c for c in COLUMNS if c not in INTEGER + ['ADR','ReservationStatusDate']]
MONTHS = {name: i for i, name in enumerate(calendar.month_name) if name}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def clean_frame(raw):
    """Return cleaned frame and audit; accepts its own output for idempotence."""
    expected = COLUMNS + (['ArrivalDate'] if 'ArrivalDate' in raw.columns else [])
    if list(raw.columns) != expected:
        raise ValueError(f'Invalid schema: expected {expected}; got {list(raw.columns)}')
    raw = raw.astype('string')
    exact_duplicates = int(raw.duplicated().sum())
    raw_missing = {c: int((raw[c].isna() | raw[c].isin(['', 'NULL', 'NA'])).sum()) for c in COLUMNS}
    frame = raw[COLUMNS].copy()
    for c in COLUMNS:
        frame[c] = frame[c].str.strip().replace({'': pd.NA, 'NULL': pd.NA, 'NA': pd.NA})
    normalized_missing = {c: int(frame[c].isna().sum()) for c in COLUMNS}
    for c in ['Agent', 'Company']:
        frame[c] = frame[c].fillna('NotApplicable')
    normalized_duplicates = int(frame.duplicated().sum())
    for c in INTEGER + ['ADR']:
        try:
            values = pd.to_numeric(frame[c], errors='raise')
            finite = values.dropna().astype(float)
            if not np.isfinite(finite).all():
                raise ValueError('non-finite values')
            if c in INTEGER:
                if ((finite % 1) != 0).any():
                    raise ValueError('non-integer values')
                frame[c] = values.astype('Int64')
            else:
                frame[c] = values.astype('Float64')
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f'Invalid numeric value in {c}: {exc}') from exc
    try:
        if frame[['ArrivalDateYear','ArrivalDateMonth','ArrivalDateDayOfMonth']].isna().any().any():
            raise ValueError('missing arrival date component')
        months = frame['ArrivalDateMonth'].map(MONTHS)
        if months.isna().any():
            raise ValueError('unknown arrival month')
        arrival = pd.to_datetime(dict(year=frame['ArrivalDateYear'].astype(int), month=months.astype(int), day=frame['ArrivalDateDayOfMonth'].astype(int)), errors='raise')
        frame['ArrivalDate'] = arrival.dt.strftime('%Y-%m-%d').astype('string')
        dates = pd.to_datetime(frame['ReservationStatusDate'], format='%Y-%m-%d', errors='raise')
        if dates.isna().any():
            raise ValueError('missing reservation status date')
        frame['ReservationStatusDate'] = dates.dt.strftime('%Y-%m-%d').astype('string')
        if 'ArrivalDate' in raw and not raw['ArrivalDate'].equals(frame['ArrivalDate']):
            raise ValueError('provided ArrivalDate does not match components')
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'Invalid date: {exc}') from exc
    stats = []
    for c in frame.columns:
        s = frame[c]
        numeric = c in INTEGER + ['ADR']
        stats.append({'column': c, 'dtype': str(s.dtype), 'raw_missing_exact': raw_missing.get(c), 'normalized_missing_before_id_mapping': normalized_missing.get(c), 'output_missing': int(s.isna().sum()), 'distinct_nonmissing': int(s.nunique()), 'min': float(s.min()) if numeric and s.notna().any() else None, 'max': float(s.max()) if numeric and s.notna().any() else None})
    flags = {'zero_total_guests': int((frame[['Adults','Children','Babies']].fillna(0).sum(axis=1).eq(0) & frame[['Adults','Children','Babies']].notna().all(axis=1)).sum()), 'adr_over_1000': int((frame['ADR'] > 1000).fillna(False).sum()), 'zero_adr': int((frame['ADR'] == 0).fillna(False).sum()), 'zero_total_nights': int(((frame['StaysInWeekendNights'] + frame['StaysInWeekNights']) == 0).fillna(False).sum()), 'undefined_categories': {c: int(frame[c].str.casefold().eq('undefined').fillna(False).sum()) for c in CATEGORICAL}}
    return frame, {'rows': len(frame), 'columns': len(frame.columns), 'raw_exact_duplicate_rows': exact_duplicates, 'post_normalization_duplicate_rows': normalized_duplicates, 'flags': flags, 'columns_profile': stats}


def run(input_path, output_path, report_dir):
    input_path, output_path, report_dir = map(Path, (input_path, output_path, report_dir))
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Input and output paths must differ')
    digest = sha256(input_path)
    raw = pd.read_csv(input_path, dtype='string', keep_default_na=False)
    clean, audit = clean_frame(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False, na_rep='', lineterminator='\n')
    audit.update({'input_file': input_path.name, 'input_sha256': digest, 'output_sha256': sha256(output_path), 'pandas_version': pd.__version__, 'numpy_version': np.__version__})
    (report_dir / 'cleaning_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pd.DataFrame(audit['columns_profile']).to_csv(report_dir / 'column_profile.csv', index=False, lineterminator='\n')
    lines = ['# Cleaning report', '', f"Input: {input_path.name}; SHA-256: `{digest}`", f"Output: {len(clean):,} rows × {len(clean.columns)} columns; SHA-256: `{audit['output_sha256']}`", f"Exact duplicate rows in raw input: {audit['raw_exact_duplicate_rows']:,}; duplicates after string normalization and ID absence mapping: {audit['post_normalization_duplicate_rows']:,}. All rows are retained.", '', 'Whitespace was trimmed; empty strings, NA and NULL (including padded markers) were normalized as missing. Agent and Company absence was mapped to NotApplicable. Numeric types and dates were validated; ArrivalDate was derived from the three arrival components. No rows were dropped and no statistics were fitted.', '', 'Flags are counts for review, not deletion rules:', '']
    lines.extend(f'- {key}: {value}' for key, value in audit['flags'].items())
    lines += ['', 'See column_profile.csv for exact raw and normalized missing counts, output missingness, cardinality, dtypes and numeric ranges. Raw missing counts recognize only exact empty strings, NA and NULL; normalized counts also recognize whitespace-padded versions. Counts of undefined categories are case-insensitive. The ADR threshold is a review heuristic, not a validity rule.']
    (report_dir / 'cleaning_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/raw/H2.csv')
    parser.add_argument('--output', default='data/processed/H2_cleaned.csv')
    parser.add_argument('--reports', default='reports')
    args = parser.parse_args()
    try:
        audit = run(args.input, args.output, args.reports)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        parser.exit(2, f'Cleaning failed: {exc}\n')
    print(f"Cleaned {audit['rows']} rows; input SHA-256 {audit['input_sha256']}")

if __name__ == '__main__':
    main()

