import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(page_title="F1 Strategy AI", page_icon="🏎️", layout="wide")

# --- STYLE ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- ASSET LOADING ---
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(__file__).resolve().parent / 'models'

@st.cache_resource
def load_assets():
    p_clf = joblib.load(MODEL_PATH / 'podium_clf.joblib')
    t_clf = joblib.load(MODEL_PATH / 'points_clf.joblib')
    reg = joblib.load(MODEL_PATH / 'pos_reg.joblib')
    df = pd.read_csv(MODEL_PATH / 'app_data_advanced.csv')
    return p_clf, t_clf, reg, df

podium_clf, points_clf, pos_reg, data = load_assets()

# --- SIDEBAR: GLOBAL CRITERIA ---
st.sidebar.image("https://www.formula1.com/etc/designs/fom-website/images/f1_logo.svg", width=150)
st.sidebar.title("Simulation Settings")

selected_circuit = st.sidebar.selectbox("🏟️ Select Grand Prix", sorted(data['race_name'].unique()))
selected_driver = st.sidebar.selectbox("👤 Select Driver", sorted(data['driverRef'].unique()))

# Filter data for the specific selection
driver_latest = data[data['driverRef'] == selected_driver].iloc[-1]
circuit_data = data[data['race_name'] == selected_circuit].iloc[0]

st.sidebar.markdown("---")
grid_pos = st.sidebar.slider("🚦 Starting Grid Position", 1, 20, 1)
quali_pos = st.sidebar.number_input("⏱️ Actual Qualifying Position", 1, 20, grid_pos)

# --- MAIN INTERFACE ---
st.title("🏎️ Formula 1 Predictive Strategy AI")
st.write(f"Analyzing **{selected_driver.upper()}** for the **{selected_circuit}**")

# Prediction Trigger
if st.sidebar.button("RUN MONTE CARLO SIMULATION", use_container_width=True):
    
    # Prepare inputs
    input_row = pd.DataFrame({
        'grid': [grid_pos],
        'driver_form': [driver_latest['driver_form']],
        'driver_track_history': [driver_latest['driver_track_history']],
        'team_form': [driver_latest['team_form']],
        'team_history': [driver_latest['team_history']],
        'circuit_passability': [circuit_data['circuit_passability']],
        'driver_points_pre_race': [driver_latest['driver_points_pre_race']],
        'driver_pos_pre_race': [driver_latest['driver_pos_pre_race']]
    })

    # Run Models
    p_podium = podium_clf.predict_proba(input_row)[0][1]
    p_points = points_clf.predict_proba(input_row)[0][1]
    expected_pos = pos_reg.predict(input_row)[0]
    expected_pos = max(1, min(20, round(expected_pos)))

    # --- TOP METRICS ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Finish", f"P{int(expected_pos)}")
    m2.metric("Podium Confidence", f"{p_podium:.1%}")
    m3.metric("Top 10 Confidence", f"{p_points:.1%}")
    m4.metric("Grid Delta", f"{grid_pos - expected_pos:+}")

    st.markdown("---")

    # --- VISUALS ---
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🏁 Outcome Probability")
        # Donut Chart for Podium Chance
        fig_prob = go.Figure(data=[go.Pie(
            labels=['Podium', 'Outside Podium'], 
            values=[p_podium, 1-p_podium], 
            hole=.6,
            marker_colors=['#e10600', '#1f1f1f']
        )])
        fig_prob.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=300)
        st.plotly_chart(fig_prob, use_container_width=True)
        st.write(f"The AI is **{p_podium:.1%}** confident in a podium finish based on current constructor ceiling and driver momentum.")

    with col_right:
        st.subheader("📈 Driver vs. Team Form")
        # Comparison Bar Chart
        comparison_df = pd.DataFrame({
            'Metric': ['Driver Avg Form', 'Team Power', 'Circuit Difficulty'],
            'Score': [driver_latest['driver_form'], driver_latest['team_form'], (1 - circuit_data['circuit_passability']) * 10]
        })
        fig_comp = px.bar(comparison_df, x='Score', y='Metric', orientation='h', 
                          color='Metric', color_discrete_sequence=px.colors.qualitative.Set1)
        fig_comp.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_comp, use_container_width=True)

    # --- DATA CONTEXT ---
    with st.expander("🔎 View Strategy Intelligence Data"):
        st.write("These are the live features being fed into the Random Forest Estimators:")
        st.dataframe(input_row)
        st.info(f"**Circuit Insight:** {selected_circuit} has an Overtake Index of {circuit_data['circuit_passability']:.2f}. "
                f"(Higher means easier to move through the field).")

else:
    # Landing state
    st.info("👈 Adjust the race criteria in the sidebar and click 'Run Simulation' to generate insights.")
    
    # Show a cool historical trend for the driver
    # --- IMPROVED HISTORICAL TREND VIZ ---
    st.subheader(f"Historical Performance Trend: {selected_driver}")

    # 1. Get the last 15 races for better context
    hist_data = data[data['driverRef'] == selected_driver].tail(15).copy()

    # 2. IMPORTANT: Create a unique label for the X-axis (Year + Race Name)
    # This prevents the "Vertical Line" issue by making every race a unique category
    hist_data['race_label'] = hist_data['year'].astype(str) + " " + hist_data['race_name']

    # 3. Create the chart using the new unique label
    fig_hist = px.line(
        hist_data, 
        x='race_label', # Use the unique label instead of just 'year'
        y='positionOrder', 
        markers=True, 
        title=f"Recent Results for {selected_driver.title()}",
        labels={'race_label': 'Grand Prix', 'positionOrder': 'Finishing Position'}
    )

    # 4. Professional Touch: Reverse Y-axis (P1 should be at the top)
    fig_hist.update_yaxes(autorange="reversed", tickmode="linear", tick0=1, dtick=2)

    # 5. Clean up X-axis labels (rotate them so they don't overlap)
    fig_hist.update_xaxes(tickangle=45, type='category') 

    st.plotly_chart(fig_hist, use_container_width=True)