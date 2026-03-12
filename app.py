"""Market Narrative Engine - Streamlit Application."""

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

from data_fetcher import fetch_price_data, fetch_news_headlines, fetch_intraday_data, fetch_company_info
from relevance_scorer import get_scorer
from download_glove import download_glove

load_dotenv()

# Download GloVe on first run
download_glove()

FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')

st.set_page_config(
    page_title="Market Narrative Engine",
    page_icon="📈",
    layout="wide"
)


@st.cache_data(ttl=5)
def get_price_data(ticker: str):
    return fetch_price_data(ticker)


@st.cache_data(ttl=5)
def get_chart_data(ticker: str, period: str, interval: str):
    return fetch_intraday_data(ticker, period=period, interval=interval)


@st.cache_data(ttl=300)
def get_news_data(ticker: str, api_key: str = '', limit: int = 10):
    return fetch_news_headlines(ticker, api_key, limit=limit)


@st.cache_data(ttl=3600)
def get_company_info(ticker: str):
    return fetch_company_info(ticker)


def calculate_delta(current: float, reference: float) -> tuple:
    if not current or not reference:
        return None, None
    delta = current - reference
    delta_pct = (delta / reference) * 100
    return delta, delta_pct


def format_price(value) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def create_chart(chart_data, ticker, chart_type, y_scale):
    fig = go.Figure()

    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=chart_data['datetime'],
            open=chart_data['Open'],
            high=chart_data['High'],
            low=chart_data['Low'],
            close=chart_data['Close'],
            name=ticker,
            increasing_line_color='#00d4aa',
            decreasing_line_color='#ff6b6b',
            increasing_fillcolor='rgba(0, 212, 170, 0.8)',
            decreasing_fillcolor='rgba(255, 107, 107, 0.8)'
        ))
    elif chart_type == 'Area':
        fig.add_trace(go.Scatter(
            x=chart_data['datetime'],
            y=chart_data['Close'],
            mode='lines',
            name=ticker,
            fill='tozeroy',
            line=dict(color='#00d4aa', width=1.5, shape='spline', smoothing=0.3),
            fillcolor='rgba(0, 212, 170, 0.15)',
            hoverinfo='x+y',
            hoverlabel=dict(font_size=10)
        ))
    else:
        fig.add_trace(go.Scatter(
            x=chart_data['datetime'],
            y=chart_data['Close'],
            mode='lines',
            name=ticker,
            line=dict(color='#00d4aa', width=1.5, shape='spline', smoothing=0.3),
            hoverinfo='x+y',
            hoverlabel=dict(font_size=10)
        ))

    price_min = chart_data['Low'].min()
    price_max = chart_data['High'].max()
    price_range = price_max - price_min
    y_min = price_min - (price_range * 0.05)
    y_max = price_max + (price_range * 0.05)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title='',
        yaxis_title='',
        showlegend=False,
        height=500,
        margin=dict(l=10, r=60, t=20, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.1)',
            gridwidth=1,
            rangeslider=dict(visible=False),
            showspikes=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.1)',
            gridwidth=1,
            side='right',
            type='log' if y_scale == 'Log' else 'linear',
            range=[y_min, y_max] if y_scale == 'Linear' else None,
            tickformat='$.2f',
            showspikes=False
        ),
        hovermode='closest',
        hoverlabel=dict(
            bgcolor='rgba(30,30,30,0.9)',
            bordercolor='rgba(255,255,255,0.1)',
            font=dict(size=11, color='rgba(255,255,255,0.7)')
        )
    )

    return fig


def main():
    st.title("📈 Market Narrative Engine")

    with st.sidebar:
        st.markdown("### Live Mode")
        live_mode = st.toggle("Enable Live Updates", value=False)
        refresh_interval = st.select_slider(
            "Refresh interval",
            options=[5, 10, 15, 30, 60],
            value=10,
            format_func=lambda x: f"{x}s"
        ) if live_mode else 10

    refresh_count = 0
    if live_mode and st.session_state.get('selected_ticker'):
        refresh_count = st_autorefresh(interval=refresh_interval * 1000, key="live_refresh")

    if 'quick_ticker' in st.session_state:
        st.session_state['selected_ticker'] = st.session_state.pop('quick_ticker')
        st.session_state['auto_fetch'] = True

    with st.form(key='ticker_form'):
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker = st.text_input(
                "Enter Ticker Symbol",
                value=st.session_state.get('selected_ticker', ''),
                placeholder="e.g., AAPL, MSFT, GOOGL",
            ).strip().upper()
        with col2:
            st.write("")
            st.write("")
            fetch_button = st.form_submit_button("🔍 Fetch Data", type="primary", width='stretch')

    if ticker:
        st.session_state['selected_ticker'] = ticker

    auto_fetch = st.session_state.pop('auto_fetch', False)
    should_fetch = (fetch_button or auto_fetch or live_mode) and ticker

    if should_fetch:
        if fetch_button:
            get_price_data.clear()
            get_chart_data.clear()

        price_data = get_price_data(ticker)

        if price_data is None:
            st.error(f"Could not fetch data for '{ticker}'. Please check the ticker symbol and try again.")
        else:
            st.session_state['price_data'] = price_data
            st.session_state['ticker'] = ticker
            st.session_state['last_update'] = datetime.now().strftime('%H:%M:%S')

            news_data = get_news_data(ticker, FINNHUB_API_KEY, 10)

            company_info = get_company_info(ticker)
            scorer = get_scorer(glove_path='data/glove.6B.50d.txt')
            scored_news = scorer.rank_articles(news_data, company_info)
            st.session_state['news_data'] = scored_news
            st.session_state['company_info'] = company_info

    if 'price_data' in st.session_state and st.session_state.get('ticker'):
        price_data = st.session_state['price_data']
        ticker = st.session_state['ticker']
        news_data = st.session_state.get('news_data', [])
        last_update = st.session_state.get('last_update', '')

        st.markdown("---")

        col_title, col_status = st.columns([3, 1])
        with col_title:
            st.subheader(f"📊 {ticker}")
        with col_status:
            if live_mode:
                st.markdown(f"🟢 **LIVE** `{last_update}`")
            else:
                st.caption("Manual refresh")

        current = price_data.get('current_price')
        open_price = price_data.get('open_price')
        delta, delta_pct = calculate_delta(current, open_price)

        col1, col2, col3 = st.columns(3)

        with col1:
            if delta is not None:
                st.metric(
                    label="Current Price",
                    value=format_price(current),
                    delta=f"{delta:+.2f} ({delta_pct:+.2f}%)"
                )
            else:
                st.metric(label="Current Price", value=format_price(current))

        with col2:
            st.metric(label="Today's High", value=format_price(price_data.get('high_price')))

        with col3:
            st.metric(label="Today's Low", value=format_price(price_data.get('low_price')))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="Open", value=format_price(price_data.get('open_price')))

        with col2:
            st.metric(label="52-Week High", value=format_price(price_data.get('week_52_high')))

        with col3:
            st.metric(label="52-Week Low", value=format_price(price_data.get('week_52_low')))

        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            chart_period = st.selectbox(
                "Period",
                options=['1d', '5d', '1mo', '3mo', '6mo', '1y'],
                index=0,
                key='chart_period'
            )
        with col2:
            if chart_period == '1d':
                interval_options = ['1m', '2m', '5m']
                default_idx = 0
            elif chart_period == '5d':
                interval_options = ['2m', '5m', '15m']
                default_idx = 1
            elif chart_period in ['1mo', '3mo']:
                interval_options = ['15m', '30m', '1h']
                default_idx = 0
            else:
                interval_options = ['1h', '1d']
                default_idx = 0

            interval = st.selectbox(
                "Interval",
                options=interval_options,
                index=default_idx,
                key='chart_interval'
            )
        with col3:
            chart_type = st.selectbox(
                "Type",
                options=['Line', 'Candlestick', 'Area'],
                index=0,
                key='chart_type'
            )
        with col4:
            y_scale = st.selectbox(
                "Scale",
                options=['Linear', 'Log'],
                index=0,
                key='y_scale'
            )

        view_options = {
            '1d': ['All', '30m', '1H', '2H', '4H'],
            '5d': ['All', '4H', '1D', '2D'],
            '1mo': ['All', '1D', '1W', '2W'],
            '3mo': ['All', '1W', '2W', '1M'],
            '6mo': ['All', '1M', '2M', '3M'],
            '1y': ['All', '1M', '3M', '6M']
        }
        current_view_options = view_options.get(chart_period, ['All'])

        if 'view_window' not in st.session_state:
            st.session_state['view_window'] = 'All'
        if st.session_state['view_window'] not in current_view_options:
            st.session_state['view_window'] = 'All'

        view_cols = st.columns(len(current_view_options))
        for i, view in enumerate(current_view_options):
            with view_cols[i]:
                is_selected = st.session_state['view_window'] == view
                if st.button(
                    view,
                    key=f"view_{view}",
                    type="primary" if is_selected else "secondary",
                    width='stretch'
                ):
                    st.session_state['view_window'] = view
                    st.rerun()

        chart_data = get_chart_data(ticker, chart_period, interval)

        if chart_data is not None and not chart_data.empty:
            view_window = st.session_state.get('view_window', 'All')
            filtered_data = chart_data.copy()

            if view_window != 'All' and len(filtered_data) > 0:
                latest_time = filtered_data['datetime'].max()

                if view_window.endswith('m'):
                    delta_time = timedelta(minutes=int(view_window[:-1]))
                elif view_window.endswith('H'):
                    delta_time = timedelta(hours=int(view_window[:-1]))
                elif view_window.endswith('D'):
                    delta_time = timedelta(days=int(view_window[:-1]))
                elif view_window.endswith('W'):
                    delta_time = timedelta(weeks=int(view_window[:-1]))
                elif view_window.endswith('M'):
                    delta_time = timedelta(days=int(view_window[:-1]) * 30)
                else:
                    delta_time = None

                if delta_time:
                    start_time = latest_time - delta_time
                    filtered_data = filtered_data[filtered_data['datetime'] >= start_time]

            if not filtered_data.empty:
                fig = create_chart(filtered_data, ticker, chart_type, y_scale)
                st.plotly_chart(fig, width='stretch', key="main_chart")
            else:
                st.warning("No data for selected time window")
        else:
            st.warning("Could not load chart data")

        st.markdown("---")
        st.subheader("📰 Latest News (sorted by relevance)")

        if not news_data:
            st.info(f"No recent news found for {ticker}.")
        else:
            for i, (article, relevance) in enumerate(news_data[:10]):
                score_pct = int(relevance.overall_score * 100)
                if score_pct >= 70:
                    indicator = "🟢"
                elif score_pct >= 40:
                    indicator = "🟡"
                else:
                    indicator = "🔴"

                breaking_badge = " 🔔 BREAKING" if relevance.is_breaking else ""

                with st.expander(f"{indicator} **{article['title']}**{breaking_badge}", expanded=(i < 3)):
                    col_score, col_meta = st.columns([2, 1])
                    with col_score:
                        st.progress(relevance.overall_score, text=f"Relevance: {score_pct}%")
                    with col_meta:
                        st.caption(f"Confidence: {int(relevance.confidence * 100)}%")

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if article.get('summary'):
                            st.write(article['summary'])
                    with col2:
                        st.caption(f"Source: {article.get('source', 'N/A')}")
                        st.caption(f"Date: {article.get('datetime', 'N/A')}")
                        if article.get('related'):
                            st.caption(f"Related: {article.get('related')}")

                    with st.popover("📊 Score Details"):
                        st.markdown("**Score Breakdown**")
                        for component, score in relevance.breakdown.items():
                            label = component.replace('_', ' ').title()
                            st.progress(score, text=f"{label}: {int(score * 100)}%")

                    if article.get('url'):
                        st.markdown(f"[Read more]({article['url']})")

    with st.sidebar:
        st.markdown("---")
        st.markdown("### Quick Tickers")
        quick_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
        for t in quick_tickers:
            if st.button(t, key=f"quick_{t}", width='stretch'):
                st.session_state['quick_ticker'] = t
                st.rerun()

        st.markdown("---")
        st.markdown("### Status")
        st.markdown("News: ✅ (yfinance)")
        if FINNHUB_API_KEY:
            st.markdown("Finnhub: ✅ (extra)")

        st.markdown("---")
        st.caption("⚠️ Data may be delayed ~15 min. Not for trading.")


if __name__ == "__main__":
    main()
