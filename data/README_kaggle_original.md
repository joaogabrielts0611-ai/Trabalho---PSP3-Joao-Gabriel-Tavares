# US Grocery and Gas Prices (2015-2026)

What everyday things actually cost in the US, month by month: a dozen eggs, a gallon
of milk, a pound of ground beef, a gallon of gas, a kilowatt-hour of electricity.

74 items, monthly, January 2015 to July 2026, from the US Bureau of Labor Statistics
**Average Price Data** programme. These are real recorded average transaction prices,
not an index -- the numbers are in dollars, so you can read them directly.

## Files

| File | Rows | Shape |
|---|---|---|
| `us_average_prices_monthly.csv` | 8,869 | **Long** -- one row per item-month. Start here. |
| `us_average_prices_wide.csv` | 139 | **Wide** -- one row per month, one column per item. Easy correlations. |
| `us_average_prices_item_summary.csv` | 74 | One row per item: coverage, min/max, and change over time. |

## Columns (long file)

| Column | Description |
|---|---|
| `date` | First day of the month the price refers to (`YYYY-MM-DD`) |
| `year`, `month` | Convenience integer columns |
| `item` | Item name, e.g. `Eggs, grade A, large` |
| `unit` | What the price is per, e.g. `doz.`, `lb. (453.6 gm)`, `gallon/3.785 liters` |
| `category` | Grouping: Beef, Fruit, Energy, Dairy and fats, ... (12 categories) |
| `price` | Average price in US dollars |
| `series_id` | Original BLS series id, so any row can be traced back to source |

The summary file adds `first_date` / `last_date` / `n_months` (coverage),
`min_price` / `max_price`, `avg_first_12mo` / `avg_last_12mo`, `pct_change_12mo_avg`,
and `is_current`.

## A starting result

Between the first 12 months and the most recent 12 months, **every one of the 60
still-published items got more expensive.** Not one fell. The median rise was **+44%**.

By category (median change): Beef **+57%**, Energy **+55%**, Vegetables **+41%**,
Bakery and grains **+27%**, Dairy **+21%**, Fruit **+16%**, Eggs **+10%**.

Coffee is the single biggest riser at **+96%** -- it roughly doubled.

## Read this before you rank anything

**1. Do not compare a single first month against a single last month.** Many of these
items are seasonal (lettuce, tomatoes, oranges), so one month is partly just the time
of year. Every change figure in the summary file is computed between **12-month
averages** at each end, which cancels the seasonality.

**2. Series start and stop at different dates.** Four items with under 24 months of
history were dropped entirely because they cannot support a trend. Of what remains,
**60 of 74 are still being published**; the other 14 stop somewhere between 2019 and
2022. `pct_change_12mo_avg` is deliberately **null for discontinued series**, so you
are never comparing a 2019 price to a 2026 one and calling it a trend. Filter with
`is_current` when you want only live series.

**3. Eggs are the cautionary tale.** Their 12-month-average change is a mild +9.5%,
but the monthly series runs from **$1.20 (June 2019) to $6.23 (March 2025)** -- a
five-fold swing driven by avian influenza. Any single-endpoint summary of eggs is
close to meaningless. This is exactly why the monthly file is the one to actually
analyse.

**4. These are nominal prices.** Nothing here is inflation-adjusted. US CPI-U rose
about 35% over this window, so an item up less than that got cheaper in real terms.

**5. National averages only.** "U.S. city average" hides very large regional
differences, and these are averages across brands, package sizes and stores.

## Good starting questions

- Which everyday items outran inflation, and which quietly got cheaper in real terms?
- How long did the 2022 price shock take to pass through to each category?
- Did the egg spike leave a permanent step up, or did prices return to trend?
- Which items move together? (The wide file makes a correlation matrix a one-liner.)
- Can gasoline prices help predict food prices a few months later?

## Source and license

US Bureau of Labor Statistics, Average Price Data (AP), U.S. city average, monthly,
not seasonally adjusted: <https://download.bls.gov/pub/time.series/ap/>

Built from `ap.series` plus `ap.data.3.Food`, `ap.data.2.Gasoline` and
`ap.data.1.HouseholdFuels`, filtered to `area_code == 0000` (national) and to series
still being published. Works of the US federal government are in the **public domain**
(CC0). BLS does not endorse this compilation.
