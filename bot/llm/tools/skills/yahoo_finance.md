---
name: get_market_prices
description: Fetch accurate closing prices and daily % change from Yahoo Finance. Use this whenever the user asks for market data, stock prices, or index levels.
metadata: {"clawhub":{"emoji":"📈","requires":{},"tools":["get_market_prices"]}}
---

# get_market_prices

Fetches real closing prices and % change directly from Yahoo Finance.
Use this as the **primary tool** for any market data question — it returns exact numbers,
unlike web_search which only returns page snippets.

## When to use

- User asks for index levels, closing prices, or daily % change
- You need specific numbers to fill in a market report
- web_search returned no numeric data in snippets

## Tool signature

```
get_market_prices(tickers: str, days: int = 5) -> str
```

## Common tickers

| Ticker     | Market             |
|------------|--------------------|
| `^TWII`    | 台灣加權指數        |
| `^GSPC`    | S&P 500            |
| `^IXIC`    | Nasdaq Composite   |
| `^SOX`     | 費半 SOX           |
| `^DJI`     | 道瓊工業指數        |
| `^VIX`     | 恐慌指數 VIX       |
| `0050.TW`  | 元大台灣50         |
| `TSMC`     | 台積電 ADR (NYSE)  |

## Usage examples

Fetch all major indices at once:
```
get_market_prices(tickers="^TWII,^GSPC,^IXIC,^SOX")
```

Fetch with extra days to cover weekends:
```
get_market_prices(tickers="^TWII,^GSPC", days=7)
```

## Output format

```
^TWII: 21543.20  +123.45 (+0.58%)  [2026-02-28]
^GSPC: 5832.10   -45.20  (-0.77%)  [2026-02-28]
^IXIC: 18421.50  -210.30 (-1.13%)  [2026-02-28]
^SOX:  4821.30   -98.40  (-2.00%)  [2026-02-28]
```

## Notes

- Always call with all needed tickers in **one call** to minimise round trips
- Use `days=7` near weekends to ensure at least 2 trading days are available for % change
- Prices are from Yahoo Finance's delayed feed (typically 15–20 min delay for live, exact for historical)
- For Taiwan stocks: append `.TW` (e.g. `2330.TW` for 台積電)