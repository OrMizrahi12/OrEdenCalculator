import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- 1. PAGE CONFIGURATION & CUSTOM CSS ---
st.set_page_config(
    page_title="Valuation Dashboard Pro",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for "Fintech" Look & Readability
st.markdown("""
<style>
    /* Global Font & Colors */
    .stApp {
        background-color: #0e1117;
        color: #ffffff; /* White Text */
    }
    
    /* Input Labels - Force White */
    label, .stTextInput label, .stNumberInput label {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 14px;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #a0a0a0;
    }
    
    /* Custom Cards */
    .result-card {
        background-color: #1e2127;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #303339;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 20px;
        height: 100%;
    }
    .result-title {
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #bbb;
        margin-bottom: 8px;
    }
    .result-value {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 4px;
        color: white;
    }
    .result-sub {
        font-size: 14px;
        font-weight: 500;
        color: #ddd;
        margin-top: 5px;
    }
    .scenario-header {
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        padding: 8px;
        border-radius: 5px;
        margin-bottom: 15px;
        color: #0e1117; /* Dark text for contrast on colored background */
    }
    
    /* Blue Stats Sidebar */
    .blue-stat {
        background-color: #1c2e4a;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 5px;
        border-left: 4px solid #2196F3;
        color: white;
    }
    .stat-label { font-size: 12px; font-weight: bold; color: #ccc; }
    .stat-value { font-size: 16px; font-weight: bold; color: #fff; }

    /* Verdict Box */
    .verdict-box {
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin-top: 10px;
    }
    .verdict-success { background-color: rgba(0, 200, 5, 0.2); border: 1px solid #00c805; color: #4caf50; }
    .verdict-danger { background-color: rgba(255, 80, 0, 0.2); border: 1px solid #ff5000; color: #ff5722; }

</style>
""", unsafe_allow_html=True)

# --- 2. TRANSLATION & DEFINITIONS DATABASE ---
TRANSLATIONS = {
    "en": {
        "title": "💎 Stock Valuation Dashboard",
        "sidebar_header": "Global Settings",
        "sidebar_stats": "Market Pulse",
        "ticker_label": "Ticker Symbol",
        "analyze_btn": "ANALYZE",
        "tab_intrinsic": "Intrinsic Value (EPS)",
        "tab_cagr": "Scenario Analysis (CAGR)",
        "current_price": "Current Price",
        "market_cap": "Market Cap",
        "pe_ratio": "P/E (TTM)",
        "avg_pe": "Hist. Avg P/E",
        "fair_value_label": "Fair Value",
        "future_price_label": "Target Price (5Y)",
        "undervalued": "UNDERVALUED",
        "overvalued": "OVERVALUED",
        "verdict": "VERDICT",
        "discount": "Discount",
        "premium": "Premium",
        "margin_safety": "Margin of Safety",
        "proj_growth": "EPS Growth Rate",
        "future_pe": "P/E (on EPS 5)",
        "discount_rate": "i (Discount Rate)",
        "rev_growth": "Rev Growth (%)",
        "net_margin": "Net Margin (%)",
        "shares_chg": "Share Change (%)",
        "pe_bear": "Bear P/E",
        "pe_base": "Base P/E",
        "pe_bull": "Bull P/E",
        "cagr_title": "Projected Annual Return (CAGR)",
        "hist_context": "Historical Performance",
        "hist_pe_chart": "P/E Ratio Trend",
        "proj_table": "Financial Projections",
        "eps_table": "EPS Growth Trajectory",
        "target_return": "Base Case Return",
        "chart_price": "Price Action (5Y)",
        "chart_financials": "Revenue & Income",
        "chart_projection": "Earnings Path",
        "hist_rev_growth": "Hist. Growth",
        "hist_net_margin": "Hist. Margin",
        "double_money": "Double Money Potential",
        "current_mkt_val": "Current Market Value",
        "years_label": "Years",
        "step1": "Step 1: Growth Assumptions",
        "step2": "Step 2: Future Financials (Year 5)",
        "step3": "Step 3: Valuation Scenarios & Target Price",
        "fut_rev": "Revenue (Yr 5)",
        "fut_ni": "Net Profit (Yr 5)",
        "fut_mcap": "Implied Market Cap",
        "fut_price": "Target Price",
        "cagr_label": "CAGR"
    },
    "he": {
        "title": "💎 לוח הערכת שווי מניות",
        "sidebar_header": "הגדרות ניתוח",
        "sidebar_stats": "תמונת מצב",
        "ticker_label": "סימול מניה",
        "analyze_btn": "בצע ניתוח",
        "tab_intrinsic": "שווי הוגן (EPS)",
        "tab_cagr": "תרחישים ותשואה (CAGR)",
        "current_price": "מחיר שוק",
        "market_cap": "שווי שוק",
        "pe_ratio": "מכפיל רווח",
        "avg_pe": "מכפיל היסטורי",
        "fair_value_label": "שווי הוגן (היום)",
        "future_price_label": "מחיר יעד (5 שנים)",
        "undervalued": "הזדמנות קנייה",
        "overvalued": "מחיר יקר",
        "verdict": "שורה תחתונה",
        "discount": "דיסקאונט",
        "premium": "פרמיה",
        "margin_safety": "מרווח ביטחון",
        "proj_growth": "צמיחת רווח (שנתית)",
        "future_pe": "P/E (על EPS 5)",
        "discount_rate": "i (קצב היוון)",
        "rev_growth": "צמיחת הכנסות (%)",
        "net_margin": "שולי רווח נקי (%)",
        "shares_chg": "שינוי מניות (שנתי)",
        "pe_bear": "מכפיל פסימי",
        "pe_base": "מכפיל בסיס",
        "pe_bull": "מכפיל אופטימי",
        "cagr_title": "צפי תשואה שנתית (CAGR)",
        "hist_context": "ביצועי עבר וממוצעים",
        "hist_pe_chart": "מגמת מכפיל רווח (P/E)",
        "proj_table": "תחזית פיננסית",
        "eps_table": "מסלול הרווח למניה (EPS)",
        "target_return": "תשואה שנתית צפויה",
        "chart_price": "תנועת מחיר (5 שנים)",
        "chart_financials": "הכנסות ורווחים",
        "chart_projection": "נתיב הרווחיות",
        "hist_rev_growth": "צמיחה היסטורית",
        "hist_net_margin": "רווחיות היסטורית",
        "double_money": "פוטנציאל הכפלה (14.4%)",
        "current_mkt_val": "שווי שוק נוכחי",
        "years_label": "שנים לתחזית",
        "step1": "שלב 1: הנחות צמיחה",
        "step2": "שלב 2: תוצאות פיננסיות (שנה 5)",
        "step3": "שלב 3: תרחישי שווי ומחיר מניה",
        "fut_rev": "הכנסות (שנה 5)",
        "fut_ni": "רווח נקי (שנה 5)",
        "fut_mcap": "שווי שוק חזוי",
        "fut_price": "מחיר מניה (2031)",
        "cagr_label": "תשואה שנתית"
    }
}

# --- 3. HELPER FUNCTIONS ---
def get_text(key, lang="en"):
    return TRANSLATIONS[lang].get(key, key)

def format_billions(value):
    if value is None: return "-"
    return f"${value / 1e9:,.2f}B"

def format_currency(value):
    if value is None: return "-"
    return f"${value:,.2f}"

def calculate_historical_pe(stock_obj, price_history):
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
        
        # Merge & Forward Fill Only (No guessing)
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
            "total_revenue_ttm": info.get("totalRevenue", 0),
            "hist_rev_cagr": hist_rev_cagr,
            "avg_net_margin": avg_net_margin,
            "history": hist,
            "hist_pe_series": hist_pe_series
        }
    except: return None

# --- 4. ADVANCED VISUALS ---

def style_chart_layout(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Roboto, sans-serif"),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#333', zeroline=False),
        hovermode="x unified"
    )
    return fig

def plot_price_history_candle(hist_df, symbol):
    hist_df['SMA50'] = hist_df['Close'].rolling(window=50).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist_df.index,
        open=hist_df['Open'], high=hist_df['High'],
        low=hist_df['Low'], close=hist_df['Close'],
        name="Price",
        increasing_line_color='#00C805', decreasing_line_color='#FF5000'
    ))
    fig.add_trace(go.Scatter(
        x=hist_df.index, y=hist_df['SMA50'], 
        name="MA 50", line=dict(color='#2196F3', width=1.5), opacity=0.7
    ))
    fig.update_layout(
        title=dict(text=f"{symbol} Price Action", font=dict(size=18, color="#ccc")),
        height=380, showlegend=False, xaxis_rangeslider_visible=False
    )
    return style_chart_layout(fig)

def plot_pe_area(pe_series, title_text):
    if pe_series is None or pe_series.empty: return go.Figure()
    fig = go.Figure(go.Scatter(
        x=pe_series.index, y=pe_series.values,
        fill='tozeroy', mode='lines',
        line=dict(color='#FFA15A', width=2),
        fillcolor='rgba(255, 161, 90, 0.1)'
    ))
    fig.update_layout(title=title_text, height=300, yaxis_title="P/E")
    return style_chart_layout(fig)

def plot_eps_trajectory_area(years, eps_data, title_text, current_year_val):
    fig = go.Figure()
    hist_x = [y for y in years if y <= current_year_val]
    hist_y = [eps_data[i] for i, y in enumerate(years) if y <= current_year_val]
    fut_x = [y for y in years if y >= current_year_val]
    fut_y = [eps_data[i] for i, y in enumerate(years) if y >= current_year_val]
    
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y, name="History",
        line=dict(color='#888', width=2), mode='lines+markers'
    ))
    fig.add_trace(go.Scatter(
        x=fut_x, y=fut_y, name="Forecast",
        fill='tozeroy', line=dict(color='#636EFA', width=3),
        fillcolor='rgba(99, 110, 250, 0.2)', mode='lines+markers'
    ))
    fig.update_layout(title=title_text, height=320, yaxis_title="EPS ($)")
    return style_chart_layout(fig)

def plot_financials_dual(df, title_text):
    if df.empty: return go.Figure()
    df = df.copy()
    if 'Total Revenue' not in df.columns: return go.Figure()
    ni_col = 'Net Income' if 'Net Income' in df.columns else 'Net Income Common Stockholders'
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index.year, y=df['Total Revenue']/1e9, 
        name='Revenue', marker_color='#2c3e50', opacity=0.8
    ))
    if ni_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index.year, y=df[ni_col]/1e9, 
            name='Net Income', line=dict(color='#00C805', width=3),
            mode='lines+markers'
        ))
    fig.update_layout(title=title_text, height=350, yaxis_title="$ Billions", barmode='overlay')
    return style_chart_layout(fig)

# --- 5. MAIN LOGIC ---

def main():
    current_year = datetime.now().year
    target_year = current_year + 5
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🌐 Language")
        lang_choice = st.radio("", ["English", "עברית"], horizontal=True, label_visibility="collapsed")
        lang = "he" if lang_choice == "עברית" else "en"
        
        st.markdown("---")
        st.subheader(get_text("sidebar_header", lang))
        ticker_input = st.text_input(get_text("ticker_label", lang), value="GOOGL").upper()
        analyze_btn = st.button(get_text("analyze_btn", lang), type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown(f"### {get_text('sidebar_stats', lang)}")
        stats_container = st.empty()
        st.caption(get_text("data_source", lang))

    # --- HEADER ---
    st.title(get_text("title", lang))

    if "stock_data" not in st.session_state: st.session_state.stock_data = None

    if analyze_btn:
        with st.spinner("Analyzing Market Data..."):
            data = fetch_stock_data(ticker_input)
            if data: st.session_state.stock_data = data
            else: st.error("Ticker not found.")

    if st.session_state.stock_data:
        data = st.session_state.stock_data
        
        # --- SIDEBAR POPULATION ---
        with stats_container.container():
            col_a, col_b = st.columns(2)
            def mini_stat(label, value):
                return f"""<div style="margin-bottom:10px;">
                        <div style="font-size:11px; color:#aaa;">{label}</div>
                        <div style="font-size:15px; font-weight:bold; color:white;">{value}</div>
                        </div>"""
            with col_a:
                st.markdown(mini_stat("P/E", f"{data['pe_ratio']:.1f}" if data['pe_ratio'] else "-"), unsafe_allow_html=True)
                st.markdown(mini_stat("EPS", f"${data['trailing_eps']}"), unsafe_allow_html=True)
            with col_b:
                st.markdown(mini_stat("Fwd P/E", f"{data['forward_pe']:.1f}" if data['forward_pe'] else "-"), unsafe_allow_html=True)
                st.markdown(mini_stat("Net Income Margin", f"{data['profit_margins']*100:.1f}%"), unsafe_allow_html=True)

        # --- HERO SECTION ---
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric(data["symbol"], format_currency(data["current_price"]))
        col_m2.metric(get_text("market_cap", lang), format_billions(data["market_cap"]))
        pe_val = f"{data['pe_ratio']:.2f}" if data['pe_ratio'] else "-"
        col_m3.metric(get_text("pe_ratio", lang), pe_val)
        col_m4.metric(get_text("avg_pe", lang), f"{data['avg_pe_5y']:.2f}" if data['avg_pe_5y'] else "-")
        
        st.plotly_chart(plot_price_history_candle(data["history"], data["symbol"]), use_container_width=True)
        
        # --- TABS ---
        st.markdown("---")
        tab1, tab2 = st.tabs([f"🔹 {get_text('tab_intrinsic', lang)}", f"🔸 {get_text('tab_cagr', lang)}"])

        # ==========================
        # CALCULATOR 1: INTRINSIC
        # ==========================
        with tab1:
            c1, c2, c3, c4 = st.columns(4)
            with c1: input_eps = st.number_input("EPS ($)", value=float(data["trailing_eps"]))
            with c2: growth_rate = st.number_input(get_text("proj_growth", lang), value=12.0) / 100
            with c3: future_pe = st.number_input(get_text("future_pe", lang), value=20.0)
            with c4: discount_rate = st.number_input(get_text("discount_rate", lang), value=10.0) / 100

            future_eps = input_eps * ((1 + growth_rate) ** 5)
            future_price = future_eps * future_pe
            fair_value = future_price / ((1 + discount_rate) ** 5)
            margin = (fair_value - data["current_price"]) / fair_value
            is_undervalued = data["current_price"] < fair_value
            
            st.write("") 
            res_c1, res_c2, res_c3 = st.columns([1, 1, 1])
            with res_c1:
                color = "#00C805" if is_undervalued else "#FF5000"
                st.markdown(f"""
                <div class="result-card" style="border-top: 4px solid {color};">
                    <div class="result-title">{get_text('fair_value_label', lang)}</div>
                    <div class="result-value" style="color: {color};">{format_currency(fair_value)}</div>
                    <div class="result-sub">Margin: {margin:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            with res_c2:
                verdict_class = "verdict-success" if is_undervalued else "verdict-danger"
                verdict_text = get_text("undervalued", lang) if is_undervalued else get_text("overvalued", lang)
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">{get_text('verdict', lang)}</div>
                    <div class="verdict-box {verdict_class}" style="font-size: 20px; margin-top:5px;">
                        {verdict_text}
                    </div>
                    <div style="margin-top:10px; font-size:12px; color:#888;">
                        Current: {format_currency(data["current_price"])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with res_c3:
                st.markdown(f"""
                <div class="result-card" style="border-top: 4px solid #636EFA;">
                    <div class="result-title">{get_text('future_price_label', lang)}</div>
                    <div class="result-value" style="color: #636EFA;">{format_currency(future_price)}</div>
                    <div class="result-sub">EPS: ${future_eps:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()
            full_years = [current_year - i for i in range(3, 0, -1)] + [current_year + i for i in range(0, 6)]
            full_years.sort()
            eps_curve = []
            for y in full_years:
                diff = y - current_year
                val = input_eps * ((1 + growth_rate) ** diff)
                eps_curve.append(val)
            
            col_chart, col_table = st.columns([2, 1])
            with col_chart:
                st.plotly_chart(plot_eps_trajectory_area(full_years, eps_curve, get_text("eps_table", lang), current_year), use_container_width=True)
            with col_table:
                st.markdown(f"**{get_text('proj_table', lang)}**")
                eps_df = pd.DataFrame({"Year": full_years, "EPS": [f"${e:.2f}" for e in eps_curve]})
                st.dataframe(eps_df, use_container_width=True, height=300)

        # ==========================
        # CALCULATOR 2: CAGR (RESTRUCTURED)
        # ==========================
        with tab2:
            st.subheader(get_text("tab_cagr", lang))
            
            # --- STEP 1: ASSUMPTIONS ---
            st.markdown(f"#### {get_text('step1', lang)}")
            col_inputs1, col_inputs2, col_inputs3, col_inputs4 = st.columns(4)
            with col_inputs1:
                rev_growth = st.number_input(get_text("rev_growth", lang), value=10.0) / 100
            with col_inputs2:
                st.markdown(f"""<div style="background-color:#1c2e4a; padding:10px; border-radius:8px; text-align:center;">
                    <span style="font-size:12px; color:#aaa;">{get_text('current_mkt_val', lang)}</span><br>
                    <span style="font-size:18px; font-weight:bold; color:white;">{format_billions(data['market_cap'])}</span>
                </div>""", unsafe_allow_html=True)
            with col_inputs3:
                st.markdown(f"""<div style="background-color:#1c2e4a; padding:10px; border-radius:8px; text-align:center;">
                    <span style="font-size:12px; color:#aaa;">{get_text('years_label', lang)}</span><br>
                    <span style="font-size:18px; font-weight:bold; color:white;">5 ({target_year})</span>
                </div>""", unsafe_allow_html=True)
            with col_inputs4:
                net_margin = st.number_input(get_text("net_margin", lang), value=20.0) / 100
            
            share_chg = st.number_input(get_text("shares_chg", lang), value=-1.0) / 100

            st.divider()

            # --- STEP 2: FINANCIALS ---
            st.markdown(f"#### {get_text('step2', lang)}")
            curr_rev = data["total_revenue_ttm"]
            fut_rev = curr_rev * ((1 + rev_growth) ** 5)
            fut_ni = fut_rev * net_margin
            
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.metric(get_text("fut_rev", lang), format_billions(fut_rev))
            with col_p2:
                st.metric(get_text("net_margin", lang), f"{net_margin*100:.1f}%")
            with col_p3:
                st.metric(get_text("fut_ni", lang), format_billions(fut_ni))

            st.divider()

            # --- STEP 3: VALUATION SCENARIOS ---
            st.markdown(f"#### {get_text('step3', lang)}")
            
            pe_col1, pe_col2, pe_col3 = st.columns(3)
            fut_shares = data["shares_outstanding"] * ((1 + share_chg) ** 5)
            if fut_shares == 0: fut_shares = 1

            scenarios = [
                {"name": "Bear", "label": get_text("pe_bear", lang), "default": 15, "color": "#EF553B"},
                {"name": "Base", "label": get_text("pe_base", lang), "default": 20, "color": "#FFA15A"},
                {"name": "Bull", "label": get_text("pe_bull", lang), "default": 25, "color": "#00C805"}
            ]

            for i, col in enumerate([pe_col1, pe_col2, pe_col3]):
                scen = scenarios[i]
                with col:
                    st.markdown(f"<div class='scenario-header' style='background-color:{scen['color']}'>{scen['name']} Case</div>", unsafe_allow_html=True)
                    
                    pe_input = st.number_input(scen["label"], value=scen["default"], key=f"pe_{i}")
                    
                    fut_mcap = fut_ni * pe_input
                    fut_price = fut_mcap / fut_shares
                    cagr = (fut_price / data["current_price"]) ** (1/5) - 1
                    
                    color_cagr = "#EF553B"
                    if cagr > 0.12: color_cagr = "#FFA15A"
                    if cagr > 0.144: color_cagr = "#00C805"
                    
                    # IMPORTANT: Fixed Indentation Here
                    st.markdown(f"""
<div class="result-card" style="border-top: 3px solid {scen['color']};">
    <div style="color: #aaa; font-size: 13px;">{get_text('fut_price', lang)}</div>
    <div style="font-size: 32px; font-weight: bold; color: white;">{format_currency(fut_price)}</div>
    <div style="margin-top: 12px; font-size: 13px; color: #aaa;">{get_text('cagr_label', lang)}</div>
    <div style="font-size: 22px; font-weight: bold; color: {color_cagr};">{cagr:.1%}</div>
    <hr style="border-color: #333;">
    <div style="font-size: 12px; color: #888;">{get_text('fut_mcap', lang)}</div>
    <div style="font-size: 14px; font-weight: bold; color: #ddd;">{format_billions(fut_mcap)}</div>
</div>
""", unsafe_allow_html=True)

    else:
        st.info("👈 Enter a ticker to begin.")

if __name__ == "__main__":
    main()