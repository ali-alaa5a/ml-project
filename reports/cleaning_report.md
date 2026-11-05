Whitespace was trimmed; empty strings, NA and NULL (including padded markers) were normalized as missing. Agent and Company absence was mapped to NotApplicable. Numeric types and dates were validated; ArrivalDate was derived from the three arrival components. No rows were dropped and no statistics were fitted.

Flags are counts for review, not deletion rules:

- zero_total_guests: 167
- adr_over_1000: 1
- zero_adr: 1208
- zero_total_nights: 331
- undefined_categories: {'ArrivalDateMonth': 0, 'Meal': 0, 'Country': 0, 'MarketSegment': 2, 'DistributionChannel': 4, 'ReservedRoomType': 0, 'AssignedRoomType': 0, 'DepositType': 0, 'Agent': 0, 'Company': 0, 'CustomerType': 0, 'ReservationStatus': 0}

See column_profile.csv for exact raw and normalized missing counts, output missingness, cardinality, dtypes and numeric ranges. Raw missing counts recognize only exact empty strings, NA and NULL; normalized counts also recognize whitespace-padded versions. Counts of undefined categories are case-insensitive. The ADR threshold is a review heuristic, not a validity rule.
