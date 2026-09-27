# Vehicle data

Obtain an authorized CSV with these columns:

| Field | Type | Meaning |
|---|---|---|
| year | numeric | Manufacturing year |
| make, model, trim, body | text | Vehicle identity/variant |
| transmission, state, color, interior | text | Other attributes |
| condition | numeric | Original scale generally 1–5 |
| odometer | numeric | Mileage in the original dataset's units |
| sellingprice | numeric | Observed auction price; training/evaluation only |

Original file: `new/Test/car_auction_train.csv`, also duplicated in `new/`. The local canonical copy is `data/car_auction_train.csv`. Original notebook display has 447,048 rows and 12 columns. Source URL, collection protocol and redistribution rights were not supplied, so CSVs are ignored and no download URL is invented.

Training filters missing make/model/target, splits 80/20 with a seed, and saves source row IDs. Additional columns are excluded. Use a genuinely independent CSV for `evaluate`; the original training CSV is not a heldout benchmark. `new/test.csv` contains cached predictions/quantiles used in historical strategy experiments; it remains local. `new/df/` is an unrelated conversion-classification dataset and is not required by the agent.
