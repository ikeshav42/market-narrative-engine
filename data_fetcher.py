"""Data fetching for stock prices and news."""

import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


def fetch_price_data(ticker: str) -> Optional[dict]:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or 'regularMarketPrice' not in info:
            hist = stock.history(period='1d')
            if hist.empty:
                return None

            current_price = hist['Close'].iloc[-1]
            open_price = hist['Open'].iloc[-1]
            high_price = hist['High'].iloc[-1]
            low_price = hist['Low'].iloc[-1]

            hist_52w = stock.history(period='1y')
            week_52_high = hist_52w['High'].max() if not hist_52w.empty else None
            week_52_low = hist_52w['Low'].min() if not hist_52w.empty else None
        else:
            current_price = info.get('regularMarketPrice') or info.get('currentPrice')
            open_price = info.get('regularMarketOpen') or info.get('open')
            high_price = info.get('regularMarketDayHigh') or info.get('dayHigh')
            low_price = info.get('regularMarketDayLow') or info.get('dayLow')
            week_52_high = info.get('fiftyTwoWeekHigh')
            week_52_low = info.get('fiftyTwoWeekLow')

        return {
            'ticker': ticker.upper(),
            'current_price': current_price,
            'open_price': open_price,
            'high_price': high_price,
            'low_price': low_price,
            'week_52_high': week_52_high,
            'week_52_low': week_52_low,
        }

    except Exception as e:
        print(f"Error fetching price data for {ticker}: {e}")
        return None


def fetch_intraday_data(ticker: str, period: str = '1d', interval: str = '5m') -> Optional[pd.DataFrame]:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            return None

        df = df.reset_index()

        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'datetime'})
        elif 'Date' in df.columns:
            df = df.rename(columns={'Date': 'datetime'})

        return df

    except Exception as e:
        print(f"Error fetching intraday data for {ticker}: {e}")
        return None


def fetch_news_yfinance(ticker: str, limit: int = 10) -> list:
    try:
        stock = yf.Ticker(ticker)
        news = stock.news

        if not news:
            return []

        headlines = []
        for item in news[:limit]:
            content = item.get('content', item)

            title = content.get('title', '') or item.get('title', '')

            provider = content.get('provider', {})
            source = provider.get('displayName', '') or item.get('publisher', 'Unknown')

            canonical = content.get('canonicalUrl', {})
            url = canonical.get('url', '') or item.get('link', '')

            pub_date = content.get('pubDate', '')
            if pub_date:
                try:
                    dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    dt_str = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    dt_str = pub_date[:16] if len(pub_date) > 16 else pub_date
            elif item.get('providerPublishTime'):
                dt_str = datetime.fromtimestamp(item.get('providerPublishTime')).strftime('%Y-%m-%d %H:%M')
            else:
                dt_str = ''

            summary = content.get('summary', '') or content.get('description', '') or ''
            if len(summary) > 200:
                summary = summary[:200] + '...'

            headlines.append({
                'title': title,
                'source': source,
                'url': url,
                'datetime': dt_str,
                'summary': summary,
                'related': ticker.upper()
            })

        return headlines

    except Exception as e:
        print(f"Error fetching yfinance news for {ticker}: {e}")
        return []


def fetch_news_finnhub(ticker: str, api_key: str, limit: int = 10) -> list:
    if not api_key:
        return []

    try:
        today = datetime.now()
        from_date = (today - timedelta(days=7)).strftime('%Y-%m-%d')
        to_date = today.strftime('%Y-%m-%d')

        url = "https://finnhub.io/api/v1/company-news"
        params = {
            'symbol': ticker.upper(),
            'from': from_date,
            'to': to_date,
            'token': api_key
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        news_data = response.json()

        if not isinstance(news_data, list):
            return []

        headlines = []
        for item in news_data[:limit]:
            headlines.append({
                'title': item.get('headline', ''),
                'source': item.get('source', ''),
                'url': item.get('url', ''),
                'datetime': datetime.fromtimestamp(item.get('datetime', 0)).strftime('%Y-%m-%d %H:%M') if item.get('datetime') else '',
                'summary': item.get('summary', '')[:200] + '...' if len(item.get('summary', '')) > 200 else item.get('summary', ''),
                'related': ticker.upper()
            })

        return headlines

    except Exception as e:
        print(f"Error fetching Finnhub news for {ticker}: {e}")
        return []


def fetch_company_info(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            'ticker': ticker.upper(),
            'shortName': info.get('shortName', ''),
            'longName': info.get('longName', ''),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'longBusinessSummary': (info.get('longBusinessSummary', '') or '')[:500]
        }
    except Exception as e:
        print(f"Error fetching company info for {ticker}: {e}")
        return {
            'ticker': ticker.upper(),
            'shortName': '',
            'longName': '',
            'sector': '',
            'industry': '',
            'longBusinessSummary': ''
        }


def fetch_news_headlines(ticker: str, api_key: str = '', limit: int = 10) -> list:
    news = fetch_news_yfinance(ticker, limit)

    if api_key and len(news) < limit:
        finnhub_news = fetch_news_finnhub(ticker, api_key, limit - len(news))
        existing_titles = {n['title'].lower() for n in news}
        for item in finnhub_news:
            if item['title'].lower() not in existing_titles:
                news.append(item)

    return news[:limit]
