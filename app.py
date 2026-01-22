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

# --- 2. TRANSLATION DATABASE ---
TRANSLATIONS = {
    "en": {
        "title": "💰 Stock Valuation Dashboard",
        "sidebar_header": "Global Settings",
        "sidebar_stats": "Key Stats",
        "ticker_label": "Enter Stock Ticker",
        "analyze_btn": "Analyze Stock",
        "data_source": "Data: Yahoo Finance",
        "tab_intrinsic": "1. Intrinsic Value (EPS)",
        "tab_cagr": "2. Scenario Analysis",
        "tab_financials": "3. Financial Statements",
        "current_price": "Current Price",
        "market_cap": "Market Cap",
        "pe_ratio": "P/E (TTM)",
        "avg_pe": "5-Year Avg P/E",
        "fair_value_label": "Fair Value (Today)",
        "future_price_label": "Target Price (Year 5)",
        "undervalued": "UNDERVALUED",
        "overvalued": "OVERVALUED",
        "verdict": "Valuation Verdict",
        "margin_safety": "Margin of Safety",
        "proj_growth": "Avg. EPS Growth Rate (%)",
        "future_pe": "P/E (on EPS 5)",
        "discount_rate": "i (Discount Rate)",
        "rev_growth": "Revenue Growth (%)",
        "net_margin": "Net Profit Margin (%)",
        "shares_chg": "Annual Shares Change (%)",
        "pe_bear": "Pessimistic P/E",
        "pe_base": "Base P/E",
        "pe_bull": "Optimistic P/E",
        "cagr_title": "Projected CAGR",
        "hist_context": "Part 1: Historical Data",
        "hist_pe_chart": "Historical P/E Ratio (5 Years)",
        "proj_table": "Financial Projections",
        "eps_table": "EPS Trajectory",
        "chart_price": "Price History",
        "chart_financials": "Revenue & Net Income",
        "step1": "Step 1: Base Assumptions (Editable)",
        "step2": "Step 2: Future Financials (Year 5)",
        "step3": "Step 3: Valuation Scenarios",
        "fut_rev": "Revenue (Yr 5)",
        "fut_ni": "Net Profit (Yr 5)",
        "fut_mcap": "Implied Market Cap",
        "fut_price": "Target Price",
        "current_mkt_val": "Current Market Cap",
        "years_label": "Years Forecast",
        "input_base_rev": "Base Revenue ($B)",
        "input_base_margin": "Base Net Margin (%)",
        "input_base_ni": "Base Net Income ($B)",
        "inc_stmt_title": "Income Statement Trends",
        "bs_title": "Balance Sheet: Assets vs. Liabilities",
        "cf_change_title": "Change in Cash Position",
        "cf_breakdown_title": "Cash Flow Breakdown (Op/Inv/Fin)",
        "metric_rev": "Total Revenue",
        "metric_gp": "Gross Profit",
        "metric_op": "Operating Income",
        "metric_ni": "Net Income",
        "assets": "Total Assets",
        "liabilities": "Total Liabilities",
        "cf_op": "Operating",
        "cf_inv": "Investing",
        "cf_fin": "Financing",
        "view_type": "View Mode",
        "view_annual": "Annual (Long Term)",
        "view_quarterly": "Quarterly (Recent)"
    },
    "he": {
        "title": "💰 לוח מכוונים להערכת שווי",
        "sidebar_header": "הגדרות כלליות",
        "sidebar_stats": "נתונים בזמן אמת",
        "ticker_label": "הכנס סימול מניה",
        "analyze_btn": "נתח מניה",
        "data_source": "מקור נתונים: Yahoo Finance",
        "tab_intrinsic": "1. ערך פנימי (EPS)",
        "tab_cagr": "2. שווי שוק ותרחישים",
        "tab_financials": "3. דוחות כספיים",
        "current_price": "מחיר נוכחי",
        "market_cap": "שווי שוק",
        "pe_ratio": "מכפיל רווח (TTM)",
        "avg_pe": "מכפיל רווח ממוצע (5 שנים)",
        "fair_value_label": "שווי הוגן (היום)",
        "future_price_label": "מחיר מניה חזוי (שנה 5)",
        "undervalued": "מתחת לשווי",
        "overvalued": "מעל השווי",
        "verdict": "פסיקת הערכה",
        "margin_safety": "מקדם ביטחון",
        "proj_growth": "קצב צמיחת רווח למניה ממוצע (%)",
        "future_pe": "P/E (על EPS 5)",
        "discount_rate": "i (קצב היוון)",
        "rev_growth": "צמיחת הכנסות (%)",
        "net_margin": "שולי רווח נקי (%)",
        "shares_chg": "שינוי מניות שנתי (%)",
        "pe_bear": "מכפיל פסימי",
        "pe_base": "מכפיל בסיס",
        "pe_bull": "מכפיל אופטימי",
        "cagr_title": "תשואה שנתית צפויה (CAGR)",
        "hist_context": "חלק ראשון: נתונים היסטוריים",
        "hist_pe_chart": "היסטוריית מכפיל רווח (5 שנים)",
        "proj_table": "חלק שני: טבלת תחזית",
        "eps_table": "מסלול רווח למניה",
        "chart_price": "היסטוריית מחיר",
        "chart_financials": "היסטוריית הכנסות ורווח נקי",
        "step1": "שלב 1: הזנת נתוני בסיס (ניתן לעריכה)",
        "step2": "שלב 2: תוצאות פיננסיות (שנה 5)",
        "step3": "שלב 3: תרחישי שווי ומחיר מניה",
        "fut_rev": "הכנסות (שנה 5)",
        "fut_ni": "רווח נקי (שנה 5)",
        "fut_mcap": "שווי שוק חזוי",
        "fut_price": "מחיר מניה חזוי (2031)",
        "current_mkt_val": "שווי שוק נוכחי",
        "years_label": "שנים לתחזית",
        "input_base_rev": "הכנסה שנתית בסיס ($B)",
        "input_base_margin": "שולי רווח בסיס (%)",
        "input_base_ni": "רווח נקי בסיס ($B)",
        "inc_stmt_title": "מגמות דוח רווח והפסד",
        "bs_title": "מאזן: נכסים מול התחייבויות",
        "cf_change_title": "שינוי במזומנים",
        "cf_breakdown_title": "פירוט תזרים מזומנים (שוטף/השקעה/מימון)",
        "metric_rev": "הכנסות",
        "metric_gp": "רווח גולמי",
        "metric_op": "רווח תפעולי",
        "metric_ni": "רווח נקי",
        "assets": "סך נכסים",
        "liabilities": "סך התחייבויות",
        "cf_op": "פעילות שוטפת",
        "cf_inv": "פעילות השקעה",
        "cf_fin": "פעילות מימון",
        "view_type": "מצב תצוגה",
        "view_annual": "שנתי (לטווח ארוך)",
        "view_quarterly": "רבעוני (עדכני להיום)"
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
    """Calculates Historical P/E using Annual data (Real Data Only)."""
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
        
        # FETCH ANNUAL (Default)
        financials = stock.financials.T.sort_index(ascending=True)
        balance_sheet = stock.balance_sheet.T.sort_index(ascending=True)
        cashflow = stock.cashflow.T.sort_index(ascending=True)
        
        # FETCH QUARTERLY (For recent data)
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
            "total_revenue_ttm": info.get("totalRevenue", 0),
            "hist_rev_cagr": hist_rev_cagr,
            "avg_net_margin": avg_net_margin,
            "history": hist,
            "hist_pe_series": hist_pe_series,
            # Data Containers
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

# --- 4. VISUALS (Standard) ---

def plot_price_history(hist_df, symbol):
    fig = px.line(hist_df, x=hist_df.index, y="Close", title=f"{symbol} Price History")
    fig.update_layout(template="plotly_dark", height=350)
    return fig

def plot_historical_pe(pe_series, title_text):
    if pe_series is None or pe_series.empty: return go.Figure()
    fig = px.line(x=pe_series.index, y=pe_series.values, title=title_text)
    fig.update_traces(line_color='#FFA15A')
    fig.update_layout(template="plotly_dark", height=300, yaxis_title="P/E")
    return fig

def plot_gauge(current_price, fair_value, title):
    bar_color = "#00CC96" if current_price < fair_value else "#EF553B"
    max_val = max(current_price, fair_value) * 1.5
    fig = go.Figure(go.Indicator(
        mode = "gauge+delta",
        value = current_price,
        domain = {'x': [0, 1], 'y': [0, 1]},
        delta = {'reference': fair_value, 'increasing': {'color': "#EF553B"}, 'decreasing': {'color': "#00CC96"}},
        gauge = {
            'axis': {'range': [0, max_val]},
            'bar': {'color': bar_color},
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': fair_value}
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), template="plotly_dark", title={'text': title})
    return fig

def plot_eps_projection(years, eps_data, title_text, current_year_val):
    fig = go.Figure()
    hist_x = [y for y in years if y <= current_year_val]
    hist_y = [eps_data[i] for i, y in enumerate(years) if y <= current_year_val]
    fut_x = [y for y in years if y >= current_year_val]
    fut_y = [eps_data[i] for i, y in enumerate(years) if y >= current_year_val]
    
    fig.add_trace(go.Scatter(x=hist_x, y=hist_y, name="History", mode='lines+markers', line=dict(color='#888')))
    fig.add_trace(go.Scatter(x=fut_x, y=fut_y, name="Projection", mode='lines+markers', line=dict(color='#636EFA')))
    fig.update_layout(title=title_text, height=350, template="plotly_dark")
    return fig

def plot_financials(df, title_text):
    if df.empty: return go.Figure()
    df = df.copy()
    if 'Total Revenue' not in df.columns: return go.Figure()
    ni_col = 'Net Income' if 'Net Income' in df.columns else 'Net Income Common Stockholders'
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index.year, y=df['Total Revenue']/1e9, name='Revenue', marker_color='#636EFA'))
    if ni_col in df.columns:
        fig.add_trace(go.Bar(x=df.index.year, y=df[ni_col]/1e9, name='Net Income', marker_color='#EF553B'))
    fig.update_layout(title=title_text, height=350, barmode='group', template="plotly_dark")
    return fig

def plot_scenario_cagr(bear, base, bull, title_text):
    scenarios = ["Bear", "Base", "Bull"]
    values = [bear * 100, base * 100, bull * 100]
    colors = ['#EF553B', '#FFA15A', '#00CC96']
    fig = go.Figure(go.Bar(x=scenarios, y=values, text=[f"{v:.1f}%" for v in values], textposition='auto', marker_color=colors))
    fig.add_shape(type="line", x0=-0.5, y0=12, x1=2.5, y1=12, line=dict(color="yellow", width=2, dash="dash"))
    fig.add_shape(type="line", x0=-0.5, y0=14.4, x1=2.5, y1=14.4, line=dict(color="white", width=2, dash="dot"))
    fig.update_layout(title=title_text, height=300, template="plotly_dark")
    return fig

# --- NEW PLOTS FOR TAB 3 ---

def plot_income_statement(df, lang):
    """Bar chart (Histogram style) for Revenue, Gross, Operating, Net."""
    if df.empty: return go.Figure()
    fig = go.Figure()
    
    # Define mapping of keys to plot
    metrics = [
        ('Total Revenue', get_text('metric_rev', lang), '#636EFA'),
        ('Gross Profit', get_text('metric_gp', lang), '#00CC96'),
        ('Operating Income', get_text('metric_op', lang), '#FFA15A'),
        ('Net Income', get_text('metric_ni', lang), '#EF553B')
    ]
    
    # Try alternative keys for Net Income
    if 'Net Income' not in df.columns and 'Net Income Common Stockholders' in df.columns:
        metrics[-1] = ('Net Income Common Stockholders', get_text('metric_ni', lang), '#EF553B')

    x_vals = df.index if 'Quarter' not in str(type(df.index)) else df.index.astype(str)

    for col, label, color in metrics:
        if col in df.columns:
            fig.add_trace(go.Bar(x=x_vals, y=df[col]/1e9, name=label, marker_color=color))
            
    fig.update_layout(title=get_text('inc_stmt_title', lang), height=400, barmode='group', template="plotly_dark", yaxis_title="Billions ($)")
    return fig

def plot_balance_sheet(df, lang):
    """Bar chart comparing Assets vs Liabilities."""
    if df.empty: return go.Figure()
    
    assets_col = 'Total Assets'
    liab_col = 'Total Liabilities Net Minority Interest'
    if liab_col not in df.columns: liab_col = 'Total Liabilities' # Fallback
    
    x_vals = df.index if 'Quarter' not in str(type(df.index)) else df.index.astype(str)
    
    fig = go.Figure()
    if assets_col in df.columns:
        fig.add_trace(go.Bar(x=x_vals, y=df[assets_col]/1e9, name=get_text('assets', lang), marker_color='#636EFA'))
    if liab_col in df.columns:
        fig.add_trace(go.Bar(x=x_vals, y=df[liab_col]/1e9, name=get_text('liabilities', lang), marker_color='#EF553B'))
        
    fig.update_layout(title=get_text('bs_title', lang), height=400, barmode='group', template="plotly_dark", yaxis_title="Billions ($)")
    return fig

def plot_cash_change(df, lang):
    """Bar chart for Changes In Cash."""
    if df.empty: return go.Figure()
    
    col = None
    for c in ['Changes In Cash', 'Change In Cash']:
        if c in df.columns: col = c; break
        
    if not col: return go.Figure()
    
    x_vals = df.index if 'Quarter' not in str(type(df.index)) else df.index.astype(str)
    
    fig = go.Figure(go.Bar(x=x_vals, y=df[col]/1e9, marker_color='#00CC96'))
    fig.update_layout(title=get_text('cf_change_title', lang), height=350, template="plotly_dark", yaxis_title="Billions ($)")
    return fig

def plot_cashflow_breakdown(df, lang):
    """Grouped bar for Op, Inv, Fin Activities."""
    if df.empty: return go.Figure()
    
    fig = go.Figure()
    map_cf = [
        ('Operating Cash Flow', get_text('cf_op', lang), '#636EFA'),
        ('Investing Cash Flow', get_text('cf_inv', lang), '#FFA15A'),
        ('Financing Cash Flow', get_text('cf_fin', lang), '#EF553B')
    ]
    
    x_vals = df.index if 'Quarter' not in str(type(df.index)) else df.index.astype(str)
    
    for col, label, color in map_cf:
        if col in df.columns:
            fig.add_trace(go.Bar(x=x_vals, y=df[col]/1e9, name=label, marker_color=color))
            
    fig.update_layout(title=get_text('cf_breakdown_title', lang), height=400, barmode='group', template="plotly_dark", yaxis_title="Billions ($)")
    return fig

# --- 5. MAIN LOGIC ---

def main():
    current_year = datetime.now().year
    target_year = current_year + 5
    
    with st.sidebar:
        st.header("🌐 Language / שפה")
        lang_choice = st.radio("Select Language", ["English", "עברית"], horizontal=True)
        lang = "he" if lang_choice == "עברית" else "en"
        
        st.divider()
        st.subheader(get_text("sidebar_header", lang))
        ticker_input = st.text_input(get_text("ticker_label", lang), value="GOOGL").upper()
        analyze_btn = st.button(get_text("analyze_btn", lang), type="primary")
        
        st.divider()
        st.subheader(get_text("sidebar_stats", lang))
        stats_container = st.container()
        st.caption(get_text("data_source", lang))

    st.title(get_text("title", lang))

    if "stock_data" not in st.session_state: st.session_state.stock_data = None

    if analyze_btn:
        with st.spinner("Loading..."):
            data = fetch_stock_data(ticker_input)
            if data: st.session_state.stock_data = data
            else: st.error("Ticker not found")

    if st.session_state.stock_data:
        data = st.session_state.stock_data
        
        # Sidebar Stats
        with stats_container:
            st.metric("P/E (TTM)", f"{data['pe_ratio']:.2f}" if data['pe_ratio'] else "-")
            st.metric("Fwd P/E", f"{data['forward_pe']:.2f}" if data['forward_pe'] else "-")
            st.metric("EPS (TTM)", f"${data['trailing_eps']}")
            st.metric("Net Income Margin", f"{data['profit_margins']*100:.2f}%")

        # Top Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stock", data["symbol"])
        c2.metric(get_text("current_price", lang), format_currency(data["current_price"]))
        c3.metric(get_text("pe_ratio", lang), f"{data['pe_ratio']:.2f}" if data['pe_ratio'] else "-")
        c4.metric(get_text("market_cap", lang), format_billions(data["market_cap"]))
        
        st.plotly_chart(plot_price_history(data["history"], data["symbol"]), use_container_width=True)
        
        # TABS
        tab1, tab2, tab3 = st.tabs([
            get_text("tab_intrinsic", lang), 
            get_text("tab_cagr", lang),
            get_text("tab_financials", lang)
        ])

        # --- TAB 1: Intrinsic Value ---
        with tab1:
            st.subheader(get_text("tab_intrinsic", lang))
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: input_eps = st.number_input("EPS ($)", value=float(data["trailing_eps"]))
            with col2: growth_rate = st.number_input(get_text("proj_growth", lang), value=12.0) / 100
            with col3: future_pe = st.number_input(get_text("future_pe", lang), value=20.0)
            with col4: discount_rate = st.number_input(get_text("discount_rate", lang), value=10.0) / 100

            future_eps = input_eps * ((1 + growth_rate) ** 5)
            future_price = future_eps * future_pe
            fair_value = future_price / ((1 + discount_rate) ** 5)
            margin = (fair_value - data["current_price"]) / fair_value
            is_undervalued = data["current_price"] < fair_value
            
            st.divider()
            
            r1, r2, r3 = st.columns(3)
            with r1:
                st.plotly_chart(plot_gauge(data["current_price"], fair_value, get_text("fair_value_label", lang)), use_container_width=True)
            with r2:
                st.metric(get_text("fair_value_label", lang), format_currency(fair_value), f"Margin: {margin:.1%}")
                if is_undervalued: st.success(f"✅ {get_text('undervalued', lang)}")
                else: st.error(f"❌ {get_text('overvalued', lang)}")
            with r3:
                st.metric(get_text("future_price_label", lang), format_currency(future_price))
                st.caption(f"Based on EPS: ${future_eps:.2f}")

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
                st.plotly_chart(plot_eps_projection(full_years, eps_curve, get_text("eps_table", lang), current_year), use_container_width=True)
            with col_table:
                st.write(f"**{get_text('proj_table', lang)}**")
                eps_df = pd.DataFrame({"Year": full_years, "EPS": [f"${e:.2f}" for e in eps_curve]})
                st.dataframe(eps_df, use_container_width=True)

        # --- TAB 2: CAGR ---
        with tab2:
            st.subheader(get_text("tab_cagr", lang))
            
            st.write(f"#### {get_text('hist_context', lang)}")
            h1, h2, h3 = st.columns(3)
            with h1: st.metric("Hist. Rev CAGR", f"{data['hist_rev_cagr']:.1%}")
            with h2: st.metric("Hist. Net Margin", f"{data['avg_net_margin']:.1%}")
            with h3: st.metric("Hist. Avg P/E", f"{data['avg_pe_5y']:.2f}")
            
            st.plotly_chart(plot_financials(data["annual"]["financials"].tail(5), get_text("chart_financials", lang)), use_container_width=True)
            
            if data['hist_pe_series'] is not None:
                st.plotly_chart(plot_historical_pe(data['hist_pe_series'], get_text("hist_pe_chart", lang)), use_container_width=True)
            else:
                st.info("Historical P/E not available")

            st.divider()

            # Step 1
            st.write(f"#### {get_text('step1', lang)}")
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                default_rev = data["total_revenue_ttm"] / 1e9 
                base_rev_input = st.number_input(get_text("input_base_rev", lang), value=default_rev)
            with col_b2:
                default_margin = data["profit_margins"] * 100
                base_margin_input = st.number_input(get_text("input_base_margin", lang), value=default_margin)
            with col_b3:
                base_net_income = base_rev_input * (base_margin_input / 100)
                st.metric(get_text("input_base_ni", lang), f"${base_net_income:,.2f}B")

            c1, c2, c3 = st.columns(3)
            with c1: rev_growth = st.number_input(get_text("rev_growth", lang), value=10.0) / 100
            with c2: share_chg = st.number_input(get_text("shares_chg", lang), value=-1.0) / 100
            with c3: st.metric(get_text("years_label", lang), f"5 ({target_year})")

            st.divider()

            # Step 2
            st.write(f"#### {get_text('step2', lang)}")
            fut_rev = (base_rev_input * 1e9) * ((1 + rev_growth) ** 5)
            fut_ni = fut_rev * (base_margin_input / 100)
            
            f1, f2, f3 = st.columns(3)
            f1.metric(get_text("fut_rev", lang), format_billions(fut_rev))
            f2.metric("Margin", f"{base_margin_input:.1f}%")
            f3.metric(get_text("fut_ni", lang), format_billions(fut_ni))

            st.divider()

            # Step 3
            st.write(f"#### {get_text('step3', lang)}")
            pe_col1, pe_col2, pe_col3 = st.columns(3)
            fut_shares = data["shares_outstanding"] * ((1 + share_chg) ** 5)
            if fut_shares == 0: fut_shares = 1
            
            scenarios = [
                (get_text("pe_bear", lang), 15),
                (get_text("pe_base", lang), 20),
                (get_text("pe_bull", lang), 25)
            ]
            cagrs = []
            for i, (label, default_val) in enumerate(scenarios):
                with [pe_col1, pe_col2, pe_col3][i]:
                    pe_val = st.number_input(label, value=default_val, key=f"pe_{i}")
                    fut_mcap = fut_ni * pe_val
                    fut_price = fut_mcap / fut_shares
                    cagr = (fut_price / data["current_price"]) ** (1/5) - 1
                    cagrs.append(cagr)
                    st.metric(get_text("fut_mcap", lang), format_billions(fut_mcap))
                    st.metric(get_text("fut_price", lang), format_currency(fut_price))
                    st.metric("CAGR", f"{cagr:.1%}")

            st.divider()
            st.plotly_chart(plot_scenario_cagr(cagrs[0], cagrs[1], cagrs[2], get_text("cagr_title", lang)), use_container_width=True)

        # --- TAB 3: FINANCIAL STATEMENTS ---
        with tab3:
            st.subheader(get_text("tab_financials", lang))
            
            # View Selector
            view_mode = st.radio(get_text("view_type", lang), [get_text("view_annual", lang), get_text("view_quarterly", lang)], horizontal=True)
            
            # Select Data Source based on toggle
            selected_data = data["annual"] if view_mode == get_text("view_annual", lang) else data["quarterly"]
            
            # 1. Income Statement
            st.plotly_chart(plot_income_statement(selected_data['financials'], lang), use_container_width=True)
            st.divider()
            
            # 2. Balance Sheet
            st.plotly_chart(plot_balance_sheet(selected_data['balance_sheet'], lang), use_container_width=True)
            st.divider()
            
            # 3. Cash Flow
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_cash_change(selected_data['cashflow'], lang), use_container_width=True)
            with c2:
                st.plotly_chart(plot_cashflow_breakdown(selected_data['cashflow'], lang), use_container_width=True)

    else:
        st.info("👈 Enter a ticker to begin.")

if __name__ == "__main__":
    main()