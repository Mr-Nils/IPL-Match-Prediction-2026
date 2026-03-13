import streamlit as st
import pickle
import pandas as pd

# Load the model and all assets
model = pickle.load(open('ipl_model_v3.pkl', 'rb'))
le_team = pickle.load(open('le_team.pkl', 'rb'))
le_toss_decision = pickle.load(open('le_toss_decision.pkl', 'rb'))
le_venue = pickle.load(open('le_venue.pkl', 'rb'))
le_match_type = pickle.load(open('le_match_type.pkl', 'rb'))
team_stats = pickle.load(open('team_stats.pkl', 'rb'))
h2h_stats = pickle.load(open('h2h_stats.pkl', 'rb'))

# 1. 2026 Specific Configuration
CURRENT_TEAMS = [
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 
    'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians', 
    'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bengaluru', 
    'Sunrisers Hyderabad'
]

TEAM_COLORS = {
    'Chennai Super Kings': '#FDB913',
    'Delhi Capitals': '#005A9C',
    'Gujarat Titans': '#1B2133',
    'Kolkata Knight Riders': '#3A225D',
    'Lucknow Super Giants': '#2980B9',
    'Mumbai Indians': '#004BA0',
    'Punjab Kings': '#ED1B24',
    'Rajasthan Royals': '#EA1B85',
    'Royal Challengers Bengaluru': '#EC1C24',
    'Sunrisers Hyderabad': '#FF822A'
}

VENUES_2026 = [
    'Arun Jaitley Stadium', 'DY Patil Stadium', 'Eden Gardens', 
    'Ekana Cricket Stadium', 'IS Bindra Stadium', 'M Chinnaswamy Stadium', 
    'MA Chidambaram Stadium', 'Narendra Modi Stadium', 
    'Rajiv Gandhi International Stadium', 'Sawai Mansingh Stadium', 'Wankhede Stadium'
]

MATCH_TYPES = ['League', 'Qualifier 1', 'Eliminator', 'Qualifier 2', 'Final']

# Streamlit Page Setup
st.set_page_config(page_title="IPL 2026 Match Predictor", layout="wide")

# Custom CSS for UI
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #0e1117; color: white; }
    .prediction-box { padding: 20px; border-radius: 10px; text-align: center; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title('🏏 IPL 2026: Match Winner Predictor')
st.markdown("Enter match details below to calculate the winning probability using historical data and team strength.")

# Sidebar Selection
st.sidebar.header("Match Settings")
match_type = st.sidebar.selectbox('Match Type', MATCH_TYPES)
venue = st.sidebar.selectbox('Venue', sorted(VENUES_2026))

# Main UI
col1, col2 = st.columns(2)

with col1:
    st.subheader("Team 1 (Home)")
    team1 = st.selectbox('Select First Team', CURRENT_TEAMS, index=0)
    color1 = TEAM_COLORS.get(team1, "#000000")
    st.markdown(f"<div style='height:10px; background-color:{color1}; border-radius:5px;'></div>", unsafe_allow_html=True)

with col2:
    st.subheader("Team 2 (Away)")
    team2 = st.selectbox('Select Second Team', CURRENT_TEAMS, index=5)
    color2 = TEAM_COLORS.get(team2, "#000000")
    st.markdown(f"<div style='height:10px; background-color:{color2}; border-radius:5px;'></div>", unsafe_allow_html=True)

st.divider()

col3, col4 = st.columns(2)
with col3:
    toss_winner = st.selectbox('Toss Winner', [team1, team2])
with col4:
    toss_decision = st.selectbox('Toss Decision', ['bat', 'field'])

# Prediction Logic
if st.button('🔥 PREDICT WINNER'):
    if team1 == team2:
        st.error("Select two different teams!")
    else:
        # Prepare Features
        t1_win_pct = team_stats.get(team1, 0.5)
        t2_win_pct = team_stats.get(team2, 0.5)
        h2h_win_pct = h2h_stats.get((team1, team2), 0.5)

        input_df = pd.DataFrame({
            'team1': [le_team.transform([team1])[0]],
            'team2': [le_team.transform([team2])[0]],
            'toss_winner': [le_team.transform([toss_winner])[0]],
            'toss_decision': [le_toss_decision.transform([toss_decision])[0]],
            'venue': [le_venue.transform([venue])[0]],
            'match_type': [le_match_type.transform([match_type])[0]],
            'team1_win_pct': [t1_win_pct],
            'team2_win_pct': [t2_win_pct],
            'h2h_win_pct': [h2h_win_pct]
        })

        # Predict
        prob = model.predict_proba(input_df)[0]
        t1_prob = prob[1]
        t2_prob = prob[0]
        
        winner = team1 if t1_prob > t2_prob else team2
        win_color = TEAM_COLORS.get(winner)
        win_prob = max(t1_prob, t2_prob)

        # Display Result
        st.balloons()
        st.markdown(f"""
            <div class="prediction-box" style="background-color:{win_color};">
                <h1>Predicted Winner: {winner}</h1>
                <h3>Confidence: {win_prob*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability Bars
        st.write("")
        st.write(f"**{team1}** probability")
        st.progress(t1_prob)
        st.write(f"**{team2}** probability")
        st.progress(t2_prob)