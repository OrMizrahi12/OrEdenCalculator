import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Stock Valuation Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- BRAND COLORS CONSTANTS ---
BRAND_RED = "#F04C24"
BRAND_YELLOW = "#FFEC4E"
BRAND_BLUE = "#4285F4"
BRAND_GREEN = "#00A859"
BRAND_BLACK = "#000000"
BRAND_WHITE = "#FFFFFF"

# --- 2. CUSTOM CSS STYLING (THEME - BLACK & BRAND COLORS) ---
st.markdown(f"""
    <style>
    /* 1. Main Background - BLACK */
    .stApp {{
        background-color: {BRAND_BLACK};
        color: {BRAND_WHITE};
    }}

    /* 2. Sidebar - BLACK with border */
    section[data-testid="stSidebar"] {{
        background-color: {BRAND_BLACK};
        border-right: 1px solid #333333;
    }}

    /* 3. Text Colors - WHITE */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {{
        color: {BRAND_WHITE} !important;
    }}
    
    /* 4. Metric Values - BRAND YELLOW */
    [data-testid="stMetricValue"] {{
        color: {BRAND_YELLOW} !important;
    }}
    /* Metric Labels - BRAND GREEN */
    [data-testid="stMetricLabel"] {{
        color: {BRAND_GREEN} !important;
    }}

    /* 5. Buttons - BRAND BLUE (Default) -> RED (Hover) */
    div.stButton > button {{
        background-color: {BRAND_BLUE};
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }}
    div.stButton > button:hover {{
        background-color: {BRAND_RED}; /* Red on Hover */
        color: white;
        border: none;
    }}

    /* 6. Input Fields - Dark Background */
    .stTextInput input, .stNumberInput input {{
        background-color: #1a1a1a;
        color: {BRAND_YELLOW};
        border: 1px solid {BRAND_BLUE};
    }}
    
    /* 7. Radio Buttons */
    div[role="radiogroup"] label {{
        color: white !important;
    }}
    
    /* 8. Tabs */
    button[data-baseweb="tab"] {{
        background-color: transparent !important;
        color: white !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        background-color: #333 !important;
        color: {BRAND_YELLOW} !important;
        border-top: 2px solid {BRAND_RED} !important;
    }}
    </style>
""", unsafe_allow_html=True)

# Brand Logo
BRAND_LOGO_URL = "https://i.ibb.co/G4Qyj4b4/The-Quant-Investors-2.png"

# --- 3. HELPER FUNCTIONS ---

def format_billions(value):
    if value is None: return "-"
    return f"${value / 1e9:,.2f}B"

def format_currency(value):
    if value is None: return "-"
    return f"${value:,.2f}"

def calculate_historical_pe(stock_obj, price_history):
    """Calculates Historical P/E using Annual data."""
    try:
        financials = stock_obj.financials.T
        if financials.empty:
            financials = stock_obj.quarterly_financials.T
        if financials.empty: return None
            
        eps_col = None
        for col in ['Basic EPS', 'Diluted EPS', 'Earnings Per Share']:
            if col in financials.columns:
                eps_col = col
                break
        if not eps_col: return None
            
        eps_data = financials[eps_col].sort_index()
        eps_data.index = pd.to_datetime(eps_data.index).tz_localize(None)

        df_pe = pd.DataFrame(index=price_history.index.tz_localize(None))
        df_pe['Close'] = price_history['Close'].copy().tz_localize(None)
        
        # Merge & Forward Fill
        df_pe['EPS'] = eps_data.reindex(df_pe.index, method='ffill')
        df_pe['PE'] = df_pe['Close'] / df_pe['EPS']
        
        df_pe = df_pe.dropna()
        df_pe = df_pe[(df_pe['PE'] > 0) & (df_pe['PE'] < 200)]
        return df_pe['PE']
    except: return None

def fetch_stock_data(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="5y")
        if hist.empty: return None
        
        current_price = hist["Close"].iloc[-1]
        info = stock.info
        financials = stock.financials.T.sort_index(ascending=True)
        
        # Fetch NEW reports
        balance_sheet = stock.balance_sheet.T.sort_index(ascending=True)
        cashflow = stock.cashflow.T.sort_index(ascending=True)
        
        # Fetch Quarterly
        q_financials = stock.quarterly_financials.T.sort_index(ascending=True)
        q_balance_sheet = stock.quarterly_balance_sheet.T.sort_index(ascending=True)
        q_cashflow = stock.quarterly_cashflow.T.sort_index(ascending=True)
        
        hist_rev_cagr = 0
        avg_net_margin = 0
        if not financials.empty and 'Total Revenue' in financials.columns:
            try:
                s = financials['Total Revenue'].iloc[0]
                e = financials['Total Revenue'].iloc[-1]
                if s > 0: hist_rev_cagr = (e / s) ** (1/(len(financials)-1)) - 1
            except: pass
            try:
                ni = 'Net Income' if 'Net Income' in financials.columns else 'Net Income Common Stockholders'
                if ni in financials.columns: avg_net_margin = (financials[ni] / financials['Total Revenue']).mean()
            except: pass

        hist_pe_series = calculate_historical_pe(stock, hist)
        avg_pe_5y = hist_pe_series.mean() if (hist_pe_series is not None and not hist_pe_series.empty) else info.get('trailingPE', 0)

        return {
            "symbol": ticker_symbol.upper(),
            "current_price": current_price,
            "pe_ratio": info.get("trailingPE", None),
            "forward_pe": info.get("forwardPE", None),
            "avg_pe_5y": avg_pe_5y,
            "trailing_eps": info.get("trailingEps", 0),
            "profit_margins": info.get("profitMargins", 0),
            "shares_outstanding": info.get("sharesOutstanding", 0),
            "market_cap": info.get("marketCap", 0),
            "financials": financials,
            "balance_sheet": balance_sheet,
            "cashflow": cashflow,
            "total_revenue_ttm": info.get("totalRevenue", 0),
            "hist_rev_cagr": hist_rev_cagr,
            "avg_net_margin": avg_net_margin,
            "history": hist,
            "hist_pe_series": hist_pe_series,
            "annual": {
                "financials": financials,
                "balance_sheet": balance_sheet,
                "cashflow": cashflow
            },
            "quarterly": {
                "financials": q_financials,
                "balance_sheet": q_balance_sheet,
                "cashflow": q_cashflow
            }
        }
    except: return None

# --- 4. VISUALS (STRICT BRAND COLORS) ---

def plot_price_history(hist_df, symbol):
    fig = px.line(hist_df, x=hist_df.index, y="Close", title=f"{symbol} Price History")
    fig.update_traces(line_color=BRAND_BLUE)  # Brand Blue
    fig.update_layout(template="plotly_dark", height=350)
    return fig

def plot_historical_pe(pe_series):
    if pe_series is None or pe_series.empty: return go.Figure()
    fig = px.line(x=pe_series.index, y=pe_series.values, title="Historical P/E Ratio (5 Years)")
    fig.update_traces(line_color=BRAND_YELLOW) # Brand Yellow
    fig.update_layout(template="plotly_dark", height=300, yaxis_title="P/E")
    return fig

def plot_gauge(current_price, fair_value):
    # Brand Red for Overvalued, Brand Green for Undervalued
    bar_color = BRAND_GREEN if current_price < fair_value else BRAND_RED
    max_val = max(current_price, fair_value) * 1.5
    fig = go.Figure(go.Indicator(
        mode = "gauge+delta",
        value = current_price,
        domain = {'x': [0, 1], 'y': [0, 1]},
        delta = {'reference': fair_value, 'increasing': {'color': BRAND_RED}, 'decreasing': {'color': BRAND_GREEN}},
        gauge = {
            'axis': {'range': [0, max_val]},
            'bar': {'color': bar_color},
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': fair_value}
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), template="plotly_dark", title={'text': "Fair Value (Today)"})
    return fig

def plot_eps_projection(years, eps_data, current_year_val):
    fig = go.Figure()
    hist_x = [y for y in years if y <= current_year_val]
    hist_y = [eps_data[i] for i, y in enumerate(years) if y <= current_year_val]
    fut_x = [y for y in years if y >= current_year_val]
    fut_y = [eps_data[i] for i, y in enumerate(years) if y >= current_year_val]
    
    # History in White (Neutral), Projection in Brand Blue
    fig.add_trace(go.Scatter(x=hist_x, y=hist_y, name="History", mode='lines+markers', line=dict(color='white')))
    fig.add_trace(go.Scatter(x=fut_x, y=fut_y, name="Projection", mode='lines+markers', line=dict(color=BRAND_BLUE)))
    fig.update_layout(title="EPS Trajectory", height=350, template="plotly_dark")
    return fig

def plot_financials(df):
    if df.empty: return go.Figure()
    df = df.copy()
    if 'Total Revenue' not in df.columns: return go.Figure()
    ni_col = 'Net Income' if 'Net Income' in df.columns else 'Net Income Common Stockholders'
    
    fig = go.Figure()
    # Revenue = Blue, Net Income = Green
    fig.add_trace(go.Bar(x=df.index.year, y=df['Total Revenue']/1e9, name='Revenue', marker_color=BRAND_BLUE))
    if ni_col in df.columns:
        fig.add_trace(go.Bar(x=df.index.year, y=df[ni_col]/1e9, name='Net Income', marker_color=BRAND_GREEN))
    fig.update_layout(title="Revenue & Net Income", height=350, barmode='group', template="plotly_dark")
    return fig

def plot_scenario_cagr(bear, base, bull):
    scenarios = ["Bear", "Base", "Bull"]
    values = [bear * 100, base * 100, bull * 100]
    # Bear = Red, Base = Yellow, Bull = Green
    colors = [BRAND_RED, BRAND_YELLOW, BRAND_GREEN]
    fig = go.Figure(go.Bar(x=scenarios, y=values, text=[f"{v:.1f}%" for v in values], textposition='auto', marker_color=colors))
    fig.add_shape(type="line", x0=-0.5, y0=12, x1=2.5, y1=12, line=dict(color="white", width=2, dash="dash"))
    fig.add_shape(type="line", x0=-0.5, y0=14.4, x1=2.5, y1=14.4, line=dict(color="white", width=2, dash="dot"))
    fig.update_layout(title="Projected CAGR", height=300, template="plotly_dark")
    return fig

# --- PLOTS FOR TAB 3 (MULTI-COLOR LOGIC) ---

def plot_income_statement(df):
    """Bar chart with 4 distinct brand colors."""
    if df.empty: return go.Figure()
    fig = go.Figure()
    
    # 4 metrics -> 4 distinct brand colors
    metrics = [
        ('Total Revenue', 'Total Revenue', BRAND_BLUE),       # 1. Blue
        ('Gross Profit', 'Gross Profit', BRAND_GREEN),        # 2. Green
        ('Operating Income', 'Operating Income', BRAND_YELLOW), # 3. Yellow
        ('Net Income', 'Net Income', BRAND_RED)               # 4. Red (Just for distinctness)
    ]
    
    if 'Net Income' not in df.columns and 'Net Income Common Stockholders' in df.columns:
        metrics[-1] = ('Net Income Common Stockholders', 'Net Income', BRAND_RED)

    x_vals = df.index if 'Quarter' not in str(type(df.index)) else df.index.astype(str)

    for col, label, color in metrics:
        if col in df.columns:
            fig.add_trace(go.Bar(x=x_vals, y=df[col]/1e9, name=label, marker_color=color))
            
    fig.update_layout(title="Income Statement Trends", height=400, barmode='group', template="plotly_dark", yaxis_title="Billions ($)")
    return fig

def plot_balance_sheet(df):
    if df.empty: return go.Figure()
    
    assets_col = 'Total Assets'
    liab_col = 'Total Liabilities Net Minority Interest'
    if liab_col not in df.columns: liab_col = 'Total Liabilities' 
    
    x_vals = df.index if 'Quarter' not in str(type(df.index)) else df.index.astype(str)
    
    fig = go.Figure()
    
    # 3 Metrics -> 3 Distinct Colors
    # Assets = Blue
    if assets_col in df.columns:
        fig.add_trace(go.Bar(x=x_vals, y=df[assets_col]/1e9, name="Total Assets", marker_color=BRAND_BLUE))
    
    # Liabilities = Red
    if liab_col in df.columns:
        fig.add_trace(go.Bar(x=x_vals, y=df[liab_col]/1e9, name="Total Liabilities", marker_color=BRAND_RED))
    
    # Equity = Green
    if assets_col in df.columns and liab_col in df.columns:
        equity_series = df[assets_col] - df[liab_col]
        fig.add_trace(go.Bar(x=x_vals, y=equity_series/1e9, name="Total Equity", marker_color=BRAND_GREEN))
        
    fig.update_layout(title="Balance Sheet: Assets, Liabilities & Equity", height=400, barmode='group', template="plotly_dark", yaxis_title="Billions ($)")
    return fig

def plot_financial_leverage(df):
    """Leverage Ratio."""
    if df.empty: return go.Figure()
    
    assets_col = 'Total Assets'
    liab_col = 'Total Liabilities Net Minority Interest'
    if liab_col not in df.columns: liab_col = 'Total Liabilities' 
    
    if assets_col not in df.columns or liab_col not in df.columns:
        return go.Figure()
    
    equity_series = df[assets_col] - df[liab_col]
    leverage_series = df[liab_col] / equity_series.replace(0, 1)
    
    x_vals = df.index if 'Quarter' not in str(type(df.index)) else df.index.astype(str)
    
    fig = go.Figure()
    # Single metric -> Use Yellow to contrast with the Blue/Red/Green above
    fig.add_trace(go.Bar(
        x=x_vals, 
        y=leverage_series, 
        name="Leverage Ratio", 
        marker_color=BRAND_YELLOW 
    ))
    
    fig.update_layout(
        title="Financial Leverage (Liabilities / Equity)", 
        height=350, 
        template="plotly_dark", 
        yaxis_title="Ratio"
    )
    return fig

def plot_cash_change(df):
    if df.empty: return go.Figure()
    
    col = None
    for c in ['Changes In Cash', 'Change In Cash']:
        if c in df.columns: col = c; break
        
    if not col: return go.Figure()
    
    x_vals = df.index if 'Quarter' not in str(type(df.index)) else df.index.astype(str)
    
    # Single metric -> Green
    fig = go.Figure(go.Bar(x=x_vals, y=df[col]/1e9, marker_color=BRAND_GREEN))
    fig.update_layout(title="Change in Cash Position", height=350, template="plotly_dark", yaxis_title="Billions ($)")
    return fig

def plot_cashflow_breakdown(df):
    if df.empty: return go.Figure()
    
    fig = go.Figure()
    # 3 Metrics -> 3 Distinct Colors
    map_cf = [
        ('Operating Cash Flow', 'Operating', BRAND_BLUE),   # 1. Blue
        ('Investing Cash Flow', 'Investing', BRAND_YELLOW), # 2. Yellow
        ('Financing Cash Flow', 'Financing', BRAND_RED)     # 3. Red
    ]
    
    x_vals = df.index if 'Quarter' not in str(type(df.index)) else df.index.astype(str)
    
    for col, label, color in map_cf:
        if col in df.columns:
            fig.add_trace(go.Bar(x=x_vals, y=df[col]/1e9, name=label, marker_color=color))
            
    fig.update_layout(title="Cash Flow Breakdown (Op/Inv/Fin)", height=400, barmode='group', template="plotly_dark", yaxis_title="Billions ($)")
    return fig

# --- 5. MAIN LOGIC ---

def main():
    current_year = datetime.now().year
    target_year = current_year + 5
    
    with st.sidebar:
        # BRAND LOGO
        st.image(BRAND_LOGO_URL, use_container_width=True)
        st.divider()
        
        st.subheader("Global Settings")
        ticker_input = st.text_input("Enter Stock Ticker", value="GOOGL").upper()
        analyze_btn = st.button("Analyze Stock", type="primary")
        
        st.divider()
        st.subheader("Key Stats")
        stats_container = st.container()
        st.caption("Data: Yahoo Finance")

    st.title("💰 Stock Valuation Dashboard")

    if "stock_data" not in st.session_state: st.session_state.stock_data = None

    if analyze_btn:
        with st.spinner("Loading..."):
            data = fetch_stock_data(ticker_input)
            if data: st.session_state.stock_data = data
            else: st.error("Ticker not found")

    if st.session_state.stock_data:
        data = st.session_state.stock_data
        
        with stats_container:
            st.metric("P/E (TTM)", f"{data['pe_ratio']:.2f}" if data['pe_ratio'] else "-")
            st.metric("Fwd P/E", f"{data['forward_pe']:.2f}" if data['forward_pe'] else "-")
            st.metric("EPS (TTM)", f"${data['trailing_eps']}")
            st.metric("Net Income Margin", f"{data['profit_margins']*100:.2f}%")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stock", data["symbol"])
        c2.metric("Current Price", format_currency(data["current_price"]))
        c3.metric("P/E (TTM)", f"{data['pe_ratio']:.2f}" if data['pe_ratio'] else "-")
        c4.metric("Market Cap", format_billions(data["market_cap"]))
        
        st.plotly_chart(plot_price_history(data["history"], data["symbol"]), use_container_width=True)
        
        tab1, tab2, tab3 = st.tabs([
            "1. Intrinsic Value (EPS)", 
            "2. Scenario Analysis",
            "3. Financial Statements"
        ])

        # --- TAB 1 ---
        with tab1:
            st.subheader("1. Intrinsic Value (EPS)")
            col1, col2, col3, col4 = st.columns(4)
            with col1: input_eps = st.number_input("EPS ($)", value=float(data["trailing_eps"]))
            with col2: growth_rate = st.number_input("Avg. EPS Growth Rate (%)", value=12.0) / 100
            with col3: future_pe = st.number_input("P/E (on EPS 5)", value=20.0)
            with col4: discount_rate = st.number_input("i (Discount Rate)", value=10.0) / 100

            future_eps = input_eps * ((1 + growth_rate) ** 5)
            future_price = future_eps * future_pe
            fair_value = future_price / ((1 + discount_rate) ** 5)
            margin = (fair_value - data["current_price"]) / fair_value
            is_undervalued = data["current_price"] < fair_value
            
            st.divider()
            
            r1, r2, r3 = st.columns(3)
            with r1:
                st.plotly_chart(plot_gauge(data["current_price"], fair_value), use_container_width=True)
            with r2:
                st.metric("Fair Value (Today)", format_currency(fair_value), f"Margin: {margin:.1%}")
                if is_undervalued: st.success("✅ UNDERVALUED")
                else: st.error("❌ OVERVALUED")
            with r3:
                st.metric("Target Price (Year 5)", format_currency(future_price))
                st.caption(f"Based on EPS: ${future_eps:.2f}")

            st.divider()
            full_years = [current_year - i for i in range(3, 0, -1)] + [current_year + i for i in range(0, 6)]
            full_years.sort()
            eps_curve = []
            for y in full_years:
                diff = y - current_year
                val = input_eps * ((1 + growth_rate) ** diff)
                eps_curve.append(val)
                
            st.plotly_chart(plot_eps_projection(full_years, eps_curve, current_year), use_container_width=True)
            
            st.divider()
            st.write("**EPS Trajectory Table**")
            eps_df = pd.DataFrame({"Year": full_years, "EPS": [f"${e:.2f}" for e in eps_curve]})
            st.dataframe(eps_df, use_container_width=True)

        # --- TAB 2 ---
        with tab2:
            st.subheader("2. Scenario Analysis")
            
            st.write("#### Part 1: Historical Data")
            h1, h2, h3 = st.columns(3)
            with h1: st.metric("Hist. Rev CAGR", f"{data['hist_rev_cagr']:.1%}")
            with h2: st.metric("Hist. Net Margin", f"{data['avg_net_margin']:.1%}")
            with h3: st.metric("Hist. Avg P/E", f"{data['avg_pe_5y']:.2f}")
            
            st.plotly_chart(plot_financials(data["annual"]["financials"].tail(5)), use_container_width=True)
            
            if data['hist_pe_series'] is not None:
                st.plotly_chart(plot_historical_pe(data['hist_pe_series']), use_container_width=True)
            else:
                st.info("Historical P/E not available")

            st.divider()

            # Step 1
            st.write("#### Step 1: Base Assumptions (Editable)")
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                default_rev = data["total_revenue_ttm"] / 1e9 
                base_rev_input = st.number_input("Base Revenue ($B)", value=default_rev)
            with col_b2:
                default_margin = data["profit_margins"] * 100
                base_margin_input = st.number_input("Base Net Margin (%)", value=default_margin)
            with col_b3:
                base_net_income = base_rev_input * (base_margin_input / 100)
                st.metric("Base Net Income ($B)", f"${base_net_income:,.2f}B")

            c1, c2, c3 = st.columns(3)
            with c1: rev_growth = st.number_input("Revenue Growth (%)", value=10.0) / 100
            
            with c2: st.metric("Current Price", format_currency(data["current_price"]))
            
            with c3: st.metric("Years Forecast", f"5 ({target_year})")

            st.divider()

            # Step 2
            st.write("#### Step 2: Future Financials (Year 5)")
            fut_rev = (base_rev_input * 1e9) * ((1 + rev_growth) ** 5)
            fut_ni = fut_rev * (base_margin_input / 100)
            
            f1, f2, f3 = st.columns(3)
            f1.metric("Revenue (Yr 5)", format_billions(fut_rev))
            f2.metric("Margin", f"{base_margin_input:.1f}%")
            f3.metric("Net Profit (Yr 5)", format_billions(fut_ni))

            st.divider()

            # Step 3
            st.write("#### Step 3: Valuation Scenarios")
            pe_col1, pe_col2, pe_col3 = st.columns(3)
            
            fut_shares = data["shares_outstanding"] 
            if fut_shares == 0: fut_shares = 1
            
            scenarios = [
                ("Pessimistic P/E", 15),
                ("Base P/E", 20),
                ("Optimistic P/E", 25)
            ]
            cagrs = []
            for i, (label, default_val) in enumerate(scenarios):
                with [pe_col1, pe_col2, pe_col3][i]:
                    pe_val = st.number_input(label, value=default_val, key=f"pe_{i}")
                    fut_mcap = fut_ni * pe_val
                    fut_price = fut_mcap / fut_shares
                    cagr = (fut_price / data["current_price"]) ** (1/5) - 1
                    cagrs.append(cagr)
                    st.metric("Implied Market Cap", format_billions(fut_mcap))
                    st.metric("Target Price", format_currency(fut_price))
                    st.metric("CAGR", f"{cagr:.1%}")

            st.divider()
            st.plotly_chart(plot_scenario_cagr(cagrs[0], cagrs[1], cagrs[2]), use_container_width=True)

        # --- TAB 3 (FULL WIDTH GRAPHS) ---
        with tab3:
            st.subheader("3. Financial Statements")
            view_mode = st.radio("View Mode", ["Annual (Long Term)", "Quarterly (Recent)"], horizontal=True)
            selected_data = data["annual"] if view_mode == "Annual (Long Term)" else data["quarterly"]
            
            # Full width charts - no columns
            st.plotly_chart(plot_income_statement(selected_data['financials']), use_container_width=True)
            st.divider()
            
            st.plotly_chart(plot_balance_sheet(selected_data['balance_sheet']), use_container_width=True)
            st.divider()
            
            st.plotly_chart(plot_financial_leverage(selected_data['balance_sheet']), use_container_width=True)
            st.divider()
            
            st.plotly_chart(plot_cash_change(selected_data['cashflow']), use_container_width=True)
            st.divider() 
            
            st.plotly_chart(plot_cashflow_breakdown(selected_data['cashflow']), use_container_width=True)

    else:
        st.info("👈 Enter a ticker to begin.")

if __name__ == "__main__":
    main()