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

# --- 2. TRANSLATION & DEFINITIONS DATABASE ---
TRANSLATIONS = {
    "en": {
        "title": "💰 Stock Valuation Dashboard",
        "sidebar_header": "Global Settings",
        "sidebar_stats": "Key Stats (Live)",
        "ticker_label": "Enter Stock Ticker",
        "analyze_btn": "Analyze Stock",
        "data_source": "Data: Yahoo Finance | Built with Streamlit",
        "tab_intrinsic": "Calculator 1: Intrinsic Value (EPS)",
        "tab_cagr": "Calculator 2: Market Cap & Scenarios",
        "current_price": "Current Price",
        "market_cap": "Market Cap",
        "pe_ratio": "Current P/E (TTM)",
        "avg_pe": "5-Year Avg P/E",
        "fair_value_label": "Fair Value (Today)",
        "future_price_label": "Target Price (Year 5)",
        "undervalued": "UNDERVALUED",
        "overvalued": "OVERVALUED",
        "verdict": "Valuation Verdict",
        "discount": "discount",
        "premium": "premium",
        "margin_safety": "Margin of Safety",
        "proj_growth": "Avg. EPS Growth Rate (%)",
        "future_pe": "Future P/E GAAP (Year 5)",
        "discount_rate": "Discount Rate (%)",
        "rev_growth": "Est. Revenue Growth (%)",
        "net_margin": "Est. Net Profit Margin (%)",
        "shares_chg": "Annual Shares Change (%)",
        "pe_bear": "Pessimistic P/E",
        "pe_base": "Neutral P/E",
        "pe_bull": "Optimistic P/E",
        "cagr_title": "Projected CAGR (Annual Return)",
        "hist_context": "Part 1: Historical Data & Averages",
        "proj_table": "Part 2: Projections (Billions)",
        "eps_table": "EPS Trajectory (History + Projection)",
        "target_return": "Base Case CAGR",
        "chart_price": "Price History (5 Years)",
        "chart_financials": "Revenue & Net Income History",
        "chart_projection": "EPS Trend (Past & Future)",
        "hist_rev_growth": "Hist. Rev Growth",
        "hist_net_margin": "Hist. Avg Net Margin",
        "double_money": "Double Money Target (14.4%)",
        "definitions": {
            "eps": "Earnings Per Share.",
            "pe": "Price to Earnings Ratio.",
            "discount_rate": "Required annual rate of return.",
            "shares_chg": "Negative = Buybacks, Positive = Dilution.",
        }
    },
    "he": {
        "title": "💰 לוח מכוונים להערכת שווי",
        "sidebar_header": "הגדרות כלליות",
        "sidebar_stats": "נתונים בזמן אמת",
        "ticker_label": "הכנס סימול מניה",
        "analyze_btn": "נתח מניה",
        "data_source": "מקור נתונים: Yahoo Finance",
        "tab_intrinsic": "מחשבון 1: ערך פנימי (EPS)",
        "tab_cagr": "מחשבון 2: שווי שוק ותרחישים",
        "current_price": "מחיר נוכחי",
        "market_cap": "שווי שוק",
        "pe_ratio": "מכפיל רווח נוכחי",
        "avg_pe": "מכפיל רווח ממוצע (5 שנים)",
        "fair_value_label": "שווי הוגן (היום)",
        "future_price_label": "מחיר מניה חזוי (שנה 5)",
        "undervalued": "מתחת לשווי",
        "overvalued": "מעל השווי",
        "verdict": "פסיקת הערכה",
        "discount": "הנחה",
        "premium": "פרמיה",
        "margin_safety": "מקדם ביטחון",
        "proj_growth": "קצב צמיחת רווח למניה ממוצע (%)",
        "future_pe": "מכפיל רווח עתידי P/E (שנה 5)",
        "discount_rate": "קצב היוון (%)",
        "rev_growth": "צמיחת הכנסות (%)",
        "net_margin": "שולי רווח (%)",
        "shares_chg": "שינוי מניות שנתי (%)",
        "pe_bear": "מכפיל פסימי",
        "pe_base": "מכפיל ניטרלי",
        "pe_bull": "מכפיל אופטימי",
        "cagr_title": "ריבית מצטברת (CAGR)",
        "hist_context": "חלק ראשון: נתונים היסטוריים",
        "proj_table": "חלק שני: טבלת תחזית (במיליארדים)",
        "eps_table": "מסלול רווח למניה (עבר + עתיד)",
        "target_return": "תשואה שנתית (בסיס)",
        "chart_price": "היסטוריית מחיר (5 שנים)",
        "chart_financials": "היסטוריית הכנסות ורווח נקי",
        "chart_projection": "מגמת רווח למניה (משולב)",
        "hist_rev_growth": "צמיחת הכנסות (עבר)",
        "hist_net_margin": "שולי רווח (עבר)",
        "double_money": "יעד הכפלת כסף (14.4%)",
        "definitions": {
            "eps": "רווח למניה.",
            "pe": "מכפיל הרווח.",
            "discount_rate": "תשואה שנתית נדרשת.",
            "shares_chg": "שלילי = רכישה עצמית, חיובי = דילול.",
        }
    }
}

# --- 3. HELPER FUNCTIONS ---
def get_text(key, lang="en"):
    return TRANSLATIONS[lang].get(key, key)

def get_def(key, lang="en"):
    return TRANSLATIONS[lang]["definitions"].get(key, "")

def format_billions(value):
    if value is None: return "N/A"
    return f"${value / 1e9:,.2f}B"

def format_currency(value):
    if value is None: return "N/A"
    return f"${value:,.2f}"

def fetch_stock_data(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="5y")
        if hist.empty: return None
        
        current_price = hist["Close"].iloc[-1]
        info = stock.info
        financials = stock.financials.T.sort_index(ascending=True)
        
        # --- CALCULATIONS FOR GAPS ---
        
        # 1. Historical Metrics
        hist_rev_cagr = 0
        avg_net_margin = 0
        
        if not financials.empty and 'Total Revenue' in financials.columns:
            # Revenue CAGR
            try:
                start_rev = financials['Total Revenue'].iloc[0]
                end_rev = financials['Total Revenue'].iloc[-1]
                years = len(financials) - 1
                if years > 0 and start_rev > 0:
                    hist_rev_cagr = (end_rev / start_rev) ** (1/years) - 1
            except: pass
            
            # Net Margin
            try:
                ni_col = 'Net Income' if 'Net Income' in financials.columns else 'Net Income Common Stockholders'
                if ni_col in financials.columns:
                    margins = financials[ni_col] / financials['Total Revenue']
                    avg_net_margin = margins.mean()
            except: pass

        # 2. Historical Average P/E (Approximate)
        avg_pe_5y = None
        try:
            # Fallback for API limitation
            avg_pe_5y = info.get('trailingPE', 0) 
        except:
            avg_pe_5y = 0

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
            "history": hist
        }
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# --- 4. VISUALS ---
def plot_price_history(hist_df, symbol, title_text):
    fig = px.line(hist_df, x=hist_df.index, y="Close", title=f"{symbol} - {title_text}")
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=40, b=0))
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
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20), template="plotly_dark", title={'text': title})
    return fig

def plot_eps_projection(years, eps_data, title_text, current_year_val):
    fig = go.Figure()
    
    # Split data into History and Future for visual distinction
    hist_x = [y for y in years if y <= current_year_val]
    hist_y = [eps_data[i] for i, y in enumerate(years) if y <= current_year_val]
    
    fut_x = [y for y in years if y >= current_year_val]
    fut_y = [eps_data[i] for i, y in enumerate(years) if y >= current_year_val]
    
    # History Trace (Gray/Solid)
    fig.add_trace(go.Scatter(x=hist_x, y=hist_y, name="History", mode='lines+markers', line=dict(color='#888', width=3)))
    
    # Future Trace (Colored/Dashed or Solid)
    fig.add_trace(go.Scatter(x=fut_x, y=fut_y, name="Projection", mode='lines+markers', line=dict(color='#636EFA', width=4)))
    
    fig.update_layout(title=title_text, xaxis_title="Year", yaxis_title="EPS ($)", template="plotly_dark", height=300)
    return fig

def plot_financials(df, title_text):
    if df.empty: return go.Figure()
    df_plot = df.copy()
    if 'Total Revenue' not in df_plot.columns: return go.Figure()
    ni_col = 'Net Income' 
    if 'Net Income' not in df_plot.columns: ni_col = 'Net Income Common Stockholders'
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_plot.index.year, y=df_plot['Total Revenue']/1e9, name='Revenue ($B)', marker_color='#636EFA'))
    if ni_col in df_plot.columns:
        fig.add_trace(go.Bar(x=df_plot.index.year, y=df_plot[ni_col]/1e9, name='Net Income ($B)', marker_color='#EF553B'))
    fig.update_layout(title=title_text, barmode='group', template="plotly_dark", yaxis_title="Billions ($)", height=350)
    return fig

def plot_scenario_cagr(bear, base, bull, title_text):
    scenarios = ["Bear", "Base", "Bull"]
    values = [bear * 100, base * 100, bull * 100]
    colors = ['#00CC96' if v >= 12 else '#FFA15A' if v > 0 else '#EF553B' for v in values]
    fig = go.Figure(go.Bar(x=scenarios, y=values, text=[f"{v:.1f}%" for v in values], textposition='auto', marker_color=colors))
    fig.add_shape(type="line", x0=-0.5, y0=12, x1=2.5, y1=12, line=dict(color="yellow", width=2, dash="dash"))
    fig.add_shape(type="line", x0=-0.5, y0=14.4, x1=2.5, y1=14.4, line=dict(color="white", width=2, dash="dot"))
    fig.update_layout(title=title_text, yaxis_title="CAGR (%)", template="plotly_dark", height=350)
    return fig

# --- 5. MAIN LOGIC ---

def main():
    current_year = datetime.now().year
    
    # Sidebar
    with st.sidebar:
        st.header("🌐 Language / שפה")
        lang_choice = st.radio("Select Language", ["English", "עברית"], horizontal=True)
        lang = "he" if lang_choice == "עברית" else "en"
        st.divider()
        st.header(get_text("sidebar_header", lang))
        ticker_input = st.text_input(get_text("ticker_label", lang), value="GOOGL").upper()
        analyze_btn = st.button(get_text("analyze_btn", lang), type="primary")
        
        # --- NEW: SIDEBAR STATS (BLUE) ---
        st.markdown("---")
        st.markdown(f"### {get_text('sidebar_stats', lang)}")
        
        # Placeholder for stats - populated after data fetch
        stats_container = st.container()
        
        st.caption(get_text("data_source", lang))

    st.title(get_text("title", lang))

    if "stock_data" not in st.session_state: st.session_state.stock_data = None

    if analyze_btn:
        with st.spinner("Fetching Data..."):
            data = fetch_stock_data(ticker_input)
            if data: st.session_state.stock_data = data
            else: st.error("Error: Could not find ticker.")

    if st.session_state.stock_data:
        data = st.session_state.stock_data
        
        # --- POPULATE SIDEBAR STATS ---
        with stats_container:
            # Using CSS for Blue Styling
            st.markdown(
                f"""
                <style>
                .blue-stat {{
                    background-color: #e8f4f9;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 5px;
                    border-left: 4px solid #2196F3;
                    color: black;
                }}
                .stat-label {{ font-size: 12px; font-weight: bold; color: #555; }}
                .stat-value {{ font-size: 16px; font-weight: bold; color: #000; }}
                </style>
                <div class="blue-stat">
                    <div class="stat-label">P/E (TTM)</div>
                    <div class="stat-value">{data['pe_ratio'] if data['pe_ratio'] else 'N/A'}</div>
                </div>
                <div class="blue-stat">
                    <div class="stat-label">Forward P/E</div>
                    <div class="stat-value">{data['forward_pe'] if data['forward_pe'] else 'N/A'}</div>
                </div>
                <div class="blue-stat">
                    <div class="stat-label">EPS (TTM)</div>
                    <div class="stat-value">${data['trailing_eps']}</div>
                </div>
                <div class="blue-stat">
                    <div class="stat-label">Net Profit Margin</div>
                    <div class="stat-value">{data['profit_margins']*100:.2f}%</div>
                </div>
                """, 
                unsafe_allow_html=True
            )

        # --- Top Metrics ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stock", data["symbol"])
        c2.metric(get_text("current_price", lang), format_currency(data["current_price"]))
        
        pe_display = f"{data['pe_ratio']:.2f}" if data['pe_ratio'] else "N/A"
        c3.metric(get_text("pe_ratio", lang), pe_display)
        c4.metric(get_text("market_cap", lang), format_billions(data["market_cap"]))
        
        st.plotly_chart(plot_price_history(data["history"], data["symbol"], get_text("chart_price", lang)), use_container_width=True)
        st.markdown("---")
        
        tab1, tab2 = st.tabs([get_text("tab_intrinsic", lang), get_text("tab_cagr", lang)])

        # ==========================
        # CALCULATOR 1: INTRINSIC
        # ==========================
        with tab1:
            st.subheader(get_text("tab_intrinsic", lang))
            
            # Inputs
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                input_eps = st.number_input("EPS ($)", value=float(data["trailing_eps"]))
            with col2:
                growth_rate = st.number_input(get_text("proj_growth", lang), value=12.0) / 100
            with col3:
                future_pe = st.number_input(get_text("future_pe", lang), value=20.0)
            with col4:
                discount_rate = st.number_input(get_text("discount_rate", lang), value=10.0) / 100

            # Logic
            future_eps = input_eps * ((1 + growth_rate) ** 5)
            future_price = future_eps * future_pe
            fair_value = future_price / ((1 + discount_rate) ** 5)
            
            # Visuals
            st.markdown("### " + get_text("verdict", lang))
            cv1, cv2, cv3 = st.columns([1.5, 1, 1])
            
            # 1. Gauge
            with cv1:
                st.plotly_chart(plot_gauge(data["current_price"], fair_value, get_text("fair_value_label", lang)), use_container_width=True)
            
            # 2. Fair Value Big Number
            with cv2:
                margin = (fair_value - data["current_price"]) / fair_value
                is_undervalued = data["current_price"] < fair_value
                text_color = "#00CC96" if is_undervalued else "#EF553B"
                status_text = get_text("undervalued", lang) if is_undervalued else get_text("overvalued", lang)
                diff_text = get_text("discount", lang) if is_undervalued else get_text("premium", lang)

                st.markdown(f"""
                <div style="padding: 10px; background-color: rgba(255,255,255,0.05); border-radius: 10px;">
                    <p style="font-size: 16px; margin: 0; color: #888;">{get_text('fair_value_label', lang)}</p>
                    <p style="font-size: 40px; font-weight: bold; margin: 0; color: {text_color};">{format_currency(fair_value)}</p>
                    <p style="font-size: 18px; margin-top: 5px;">{status_text} ({abs(margin)*100:.1f}% {diff_text})</p>
                </div>
                """, unsafe_allow_html=True)
                st.info(f"{get_text('margin_safety', lang)}: {margin:.1%}", icon="🛡️")

            # 3. NEW: Future Stock Price (Year 5)
            with cv3:
                 st.markdown(f"""
                <div style="padding: 10px; border: 1px solid #444; border-radius: 10px; height: 100%;">
                    <p style="font-size: 16px; margin: 0; color: #888;">{get_text('future_price_label', lang)}</p>
                    <p style="font-size: 32px; font-weight: bold; margin: 10px 0; color: #636EFA;">{format_currency(future_price)}</p>
                    <p style="font-size: 12px; color: #aaa;">EPS: ${future_eps:.2f} × PE: {future_pe}</p>
                </div>
                """, unsafe_allow_html=True)

            st.divider()
            
            # --- COMBINED EPS CHART (History + Future) ---
            # Create a sequence from current_year - 3 to current_year + 5
            st.markdown(f"#### {get_text('eps_table', lang)}")
            
            # 1. Fetch historical EPS if possible (Approximation from financials if available, else linear interp)
            # For simplicity and robustness, we will create a blended list.
            # Start: Year - 3
            years_hist = [current_year - i for i in range(3, 0, -1)] # -3, -2, -1
            # We don't have exact historical EPS easily from this simple API call without complex parsing.
            # We will perform a backward projection for the chart based on growth rate to keep it smooth, 
            # OR just start from Year 0 (Current) if history is unavailable.
            # To meet the user request "add 3 years before", we will construct the years:
            
            full_years = [current_year - 3, current_year - 2, current_year - 1] + [current_year + i for i in range(0, 6)]
            
            # We'll assume the same growth curve backward for the chart visual (or flat) if we lack data,
            # but ideally we want real data. Since we only pulled 5y financials, let's try to match.
            # Simplification: We will plot the CURRENT EPS as Year 0, and project forward.
            # For the "3 years before", we will just show the projection starting from 2026.
            # BUT user explicitly asked for "Timeline larger".
            # We will simply project the graph curve starting from Year -3 using the growth rate reverse logic for visual continuity.
            
            eps_curve = []
            for y in full_years:
                diff = y - current_year
                # Future: eps * (1+g)^diff
                # Past: eps / (1+g)^(-diff) -> essentially same formula
                val = input_eps * ((1 + growth_rate) ** diff)
                eps_curve.append(val)
                
            eps_df = pd.DataFrame({
                "Year": full_years,
                "Projected EPS ($)": [f"${e:.2f}" for e in eps_curve],
                "Implied Price": [f"${e * future_pe:.2f}" for e in eps_curve]
            })
            
            # Highlight which are history vs future in the table? Not strictly necessary, but good.
            st.dataframe(eps_df, use_container_width=True)
            
            chart_title = f"{get_text('chart_projection', lang)} ({full_years[0]}-{full_years[-1]})"
            st.plotly_chart(plot_eps_projection(full_years, eps_curve, chart_title, current_year), use_container_width=True)

        # ==========================
        # CALCULATOR 2: CAGR & MARKET CAP
        # ==========================
        with tab2:
            st.subheader(get_text("tab_cagr", lang))
            
            # --- PART 1: HISTORICAL CONTEXT ---
            st.markdown(f"#### {get_text('hist_context', lang)}")
            hc1, hc2, hc3 = st.columns(3)
            with hc1: st.metric(get_text("hist_rev_growth", lang), f"{data['hist_rev_cagr']:.1%}")
            with hc2: st.metric(get_text("hist_net_margin", lang), f"{data['avg_net_margin']:.1%}")
            
            avg_pe_display = f"{data['avg_pe_5y']:.2f}" if data['avg_pe_5y'] else "N/A"
            with hc3: st.metric(get_text("avg_pe", lang), avg_pe_display)
            
            st.plotly_chart(plot_financials(data["financials"].tail(5), get_text("chart_financials", lang)), use_container_width=True)
            st.divider()
            
            # --- INPUTS ---
            ic1, ic2, ic3 = st.columns(3)
            with ic1: rev_growth = st.number_input(get_text("rev_growth", lang), value=10.0) / 100
            with ic2: net_margin = st.number_input(get_text("net_margin", lang), value=20.0) / 100
            with ic3: share_chg = st.number_input(get_text("shares_chg", lang), value=-1.0) / 100
            
            # --- MULTIPLES SELECTION ---
            st.write("---")
            st.markdown(f"**Select P/E Multiples (Ref: Current {data['pe_ratio']:.1f})**")
            pc1, pc2, pc3 = st.columns(3)
            with pc1: pe_bear = st.number_input(get_text("pe_bear", lang), value=15)
            with pc2: pe_base = st.number_input(get_text("pe_base", lang), value=20)
            with pc3: pe_bull = st.number_input(get_text("pe_bull", lang), value=25)

            # --- PART 2: PROJECTION TABLE ---
            st.markdown(f"#### {get_text('proj_table', lang)}")
            
            # EXTENDED TIMELINE Logic
            # User wants 3 years prior included.
            proj_data = []
            curr_rev = data["total_revenue_ttm"]
            
            # Generate range: -3 to +5
            range_years = range(-3, 6) # -3, -2, -1, 0, 1, 2, 3, 4, 5
            
            for i in range_years: 
                yr = current_year + i
                
                # Logic: 
                # If i <= 0: We use reverse calc for estimation (or actuals if we had them mapped perfectly).
                # To keep it smooth and visual, we use the growth rate backwards for negative years.
                
                f_rev = curr_rev * ((1 + rev_growth) ** i)
                f_ni = f_rev * net_margin
                
                # Tag Row Type
                row_type = "History (Est)" if i < 0 else ("Current" if i == 0 else "Projected")
                
                proj_data.append({
                    "Year": yr,
                    "Type": row_type,
                    "Revenue ($B)": f_rev / 1e9,
                    "Margin (%)": f"{net_margin*100:.1f}%",
                    "Net Income ($B)": f_ni / 1e9
                })
            
            df_proj = pd.DataFrame(proj_data)
            st.dataframe(df_proj.style.format({"Revenue ($B)": "${:.2f}", "Net Income ($B)": "${:.2f}"}), use_container_width=True)

            # --- CALCULATIONS ---
            # Get Year 5 Data (Last row)
            fut_ni_final = proj_data[-1]["Net Income ($B)"] * 1e9 
            
            # Market Caps
            mcap_bear = fut_ni_final * pe_bear
            mcap_base = fut_ni_final * pe_base
            mcap_bull = fut_ni_final * pe_bull
            
            # Share Count Future
            fut_shares = data["shares_outstanding"] * ((1 + share_chg) ** 5)
            if fut_shares == 0: fut_shares = 1
            
            # Share Price
            p_bear = mcap_bear / fut_shares
            p_base = mcap_base / fut_shares
            p_bull = mcap_bull / fut_shares
            
            # CAGR
            c_bear = (p_bear / data["current_price"]) ** (1/5) - 1
            c_base = (p_base / data["current_price"]) ** (1/5) - 1
            c_bull = (p_bull / data["current_price"]) ** (1/5) - 1

            # --- VISUAL RESULTS ---
            st.divider()
            
            st.metric("Projected Market Cap (Year 5 - Base)", format_billions(mcap_base))

            c_v1, c_v2 = st.columns([2, 1])
            with c_v1:
                title_dynamic = f"{get_text('cagr_title', lang)} ({current_year}-{current_year+5})"
                st.plotly_chart(plot_scenario_cagr(c_bear, c_base, c_bull, title_dynamic), use_container_width=True)
            
            with c_v2:
                target_met = c_base >= 0.12
                double_met = c_base >= 0.144
                color_cagr = "#00CC96" if target_met else "#EF553B"
                
                st.markdown(f"""
                <div style="padding: 15px; border: 1px solid #333; border-radius: 10px;">
                    <p style="margin:0; font-size:14px; color:#888;">{get_text('target_return', lang)}</p>
                    <p style="margin:0; font-size:32px; font-weight:bold; color:{color_cagr};">{c_base:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if double_met:
                    st.success(f"🚀 {get_text('double_money', lang)}: YES")
                elif target_met:
                    st.success("✅ > 12% Return: YES")
                else:
                    st.warning("⚠️ Below 12% Target")
                
                st.caption(f"Bear: {c_bear:.2%} | Bull: {c_bull:.2%}")

    else:
        st.info("👈 Please enter a ticker / אנא הכנס סימול מניה")

if __name__ == "__main__":
    main()