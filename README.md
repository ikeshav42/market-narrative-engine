# Market Narrative Engine

Real-time stock dashboard with news relevance scoring using GloVe word embeddings.

## Features

- Live stock prices with auto-refresh
- Interactive candlestick/line/area charts
- News ranked by semantic relevance to the company

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download GloVe embeddings (~862MB)
python download_glove.py

# Optional: add Finnhub API key for more news
cp .env.example .env

# Run
streamlit run app.py
```

## Relevance Scoring

Each news article is scored against the company using a weighted composite:

```
score = 0.45(direct) + 0.40(semantic) + 0.10(source) + 0.05(temporal)

if breaking_news:
    score = min(1.0, score + 0.2)
```

### Direct Match (45%)
Regex word-boundary matching for ticker symbol and company name in article text.

### Semantic Similarity (40%)
Uses GloVe word embeddings to compute similarity between article and company description.

**Text to Vector (Mean Pooling):**
```
vec(text) = (1/n) * Σ embed(word_i)
```

**Cosine Similarity:**
```
similarity = (A · B) / (||A|| × ||B||)
```

Returns value in [-1, 1], mapped to [0, 1] for scoring.

### Source Prior (10%)
Lookup table of news source reliability (Reuters: 0.95, Bloomberg: 0.95, etc.)

### Temporal Decay (5%)
Exponential decay with 6-hour half-life:
```
score = 0.5 ^ (hours_elapsed / 6)
```

## Roadmap

- [ ] News summarization using LLMs
- [ ] Price movement predictions (LSTM/transformer models)
- [ ] Sentiment analysis pipeline
- [ ] Historical data warehouse
- [ ] Correlation analysis between news sentiment and price action

## Tech Stack

- Streamlit + Plotly
- yfinance for market data
- GloVe 50d word embeddings (Stanford NLP)
