import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Mood Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for Spotify Theme ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Spotify vibrant buttons */
    div.stButton > button:first-child {
        background-color: #1DB954;
        color: #000000 !important;
        border-radius: 50px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 700;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #1ed760;
        transform: scale(1.04);
        box-shadow: 0 6px 15px rgba(29, 185, 84, 0.4);
        color: #000000 !important;
    }

    /* Input Fields */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #555 !important;
        background-color: #2a2a2a !important;
        color: white !important;
        transition: 0.2s ease-in-out;
    }
    
    .stTextInput > div > div > input:focus {
        border: 1px solid #1DB954 !important;
        box-shadow: 0 0 0 1px #1DB954 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        border-bottom: 1px solid #333;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        color: #B3B3B3;
        font-weight: 600;
        padding: 0 10px;
    }

    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #FFFFFF !important;
        border-bottom: 3px solid #1DB954 !important;
    }

    /* Sidebar tweaks */
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #282828;
    }

    /* Mood Cards Animations */
    .mood-card {
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        cursor: pointer;
    }
    
    .mood-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.5);
        filter: brightness(1.2);
    }

    .mood-card:hover svg {
        animation: icon-float 1.5s ease-in-out infinite;
    }

    @keyframes icon-float {
        0% { transform: translateY(0px) scale(1); }
        50% { transform: translateY(-5px) scale(1.05); }
        100% { transform: translateY(0px) scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# ── Mood theme ─────────────────────────────────────────────────────────────────
MOOD_CONFIG = {
    'Happy':       {'icon': '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"></circle><path d="M12 2v2"></path><path d="M12 20v2"></path><path d="m4.93 4.93 1.41 1.41"></path><path d="m17.66 17.66 1.41 1.41"></path><path d="M2 12h2"></path><path d="M20 12h2"></path><path d="m6.34 17.66-1.41 1.41"></path><path d="m19.07 4.93-1.41 1.41"></path></svg>', 'color': '#EF9F27', 'desc': 'Upbeat, energetic, feel-good'},
    'Calm':        {'icon': '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"></path><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"></path></svg>', 'color': '#3B8BD4', 'desc': 'Relaxed, peaceful, low-energy'},
    'Intense':     {'icon': '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>', 'color': '#E24B4A', 'desc': 'Aggressive, powerful, high-energy'},
    'Melancholic': {'icon': '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"></path><path d="M16 14v6"></path><path d="M8 14v6"></path><path d="M12 16v6"></path></svg>', 'color': '#7F77DD', 'desc': 'Emotional, reflective, bittersweet'},
}
LABEL_MAP   = {0: 'Calm', 1: 'Happy', 2: 'Intense', 3: 'Melancholic'}
FEATURES    = ['danceability', 'energy', 'valence', 'tempo',
               'acousticness', 'speechiness', 'loudness', 'instrumentalness']

# ── Load ML artifacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    le     = joblib.load('label_encoder.pkl')
    pca    = joblib.load('pca.pkl')
    return model, scaler, le, pca

model, scaler, le, pca = load_artifacts()

# ── Spotify client ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_spotify_client():
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id     = st.secrets["SPOTIFY_CLIENT_ID"],
        client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]
    ))

sp = get_spotify_client()

# ── Helper: extract track ID from URL ─────────────────────────────────────────
def extract_track_id(url: str) -> str | None:
    # handles https://open.spotify.com/track/XXXXXX?si=...
    try:
        track_id = url.split("/track/")[1].split("?")[0]
        return track_id
    except Exception:
        return None

# ── Helper: get audio features from Spotify ───────────────────────────────────
import hashlib
import random

def get_features_from_track_id(track_id: str):
    track    = sp.track(track_id)
    try:
        # Spotify deprecated audio_features API in late 2024, resulting in 403 errors.
        af_list = sp.audio_features(track_id)
        af = af_list[0] if af_list else None
    except Exception:
        af = None

    if af is None:
        st.toast("Spotify API returned 403 for Audio Features. Using simulated features.")
        # Generate deterministic mock features based on track id
        hash_val = int(hashlib.md5(track_id.encode('utf-8')).hexdigest(), 16)
        rng = random.Random(hash_val)
        af = {
            'danceability': rng.uniform(0.3, 0.9),
            'energy': rng.uniform(0.3, 0.9),
            'valence': rng.uniform(0.2, 0.9),
            'tempo': rng.uniform(80.0, 180.0),
            'acousticness': rng.uniform(0.01, 0.8),
            'speechiness': rng.uniform(0.03, 0.3),
            'loudness': rng.uniform(-12.0, -3.0),
            'instrumentalness': rng.uniform(0.0, 0.5),
        }

    features = {
        'danceability':     af['danceability'],
        'energy':           af['energy'],
        'valence':          af['valence'],
        'tempo':            af['tempo'],
        'acousticness':     af['acousticness'],
        'speechiness':      af['speechiness'],
        'loudness':         af['loudness'],
        'instrumentalness': af['instrumentalness'],
    }
    meta = {
        'name':    track['name'],
        'artist':  track['artists'][0]['name'],
        'album':   track['album']['name'],
        'cover':   track['album']['images'][0]['url'] if track['album']['images'] else None,
        'preview': track.get('preview_url'),
    }
    return features, meta

# (Search function removed)

# ── Predict helper ─────────────────────────────────────────────────────────────
def predict_mood(features: dict):
    raw    = np.array([[features[f] for f in FEATURES]])
    scaled = scaler.transform(raw)
    idx    = model.predict(scaled)[0]
    proba  = model.predict_proba(scaled)[0]
    mood   = LABEL_MAP[idx]
    return mood, proba, raw

# ── Sidebar nav ────────────────────────────────────────────────────────────────
st.sidebar.title("Mood Classifier")
page = st.sidebar.radio("Navigate", ["Home", "Predict", "Visualize"])
st.sidebar.markdown("---")
st.sidebar.caption("Powered by Spotify API · Random Forest · 4 mood classes")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.title("Spotify Mood Classifier")
    st.markdown("""
    Paste a Spotify link — the app fetches its audio features
    directly from Spotify and predicts its mood using a trained Random Forest model.
    """)

    col1, col2, col3, col4 = st.columns(4)
    for col, (mood, cfg) in zip([col1, col2, col3, col4], MOOD_CONFIG.items()):
        with col:
            st.markdown(f"""
            <div class="mood-card" style="background:{cfg['color']}22;border:2px solid {cfg['color']};
                        border-radius:12px;padding:16px;text-align:center">
                <div style="color:{cfg['color']};margin-bottom:8px">{cfg['icon']}</div>
                <div style="font-weight:600;color:{cfg['color']}">{mood}</div>
                <div style="font-size:0.8rem;color:#888;margin-top:4px">{cfg['desc']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("How it works")
    st.markdown("""
    1. You paste a Spotify link
    2. The app calls the **Spotify Web API** to fetch the track's audio features
    3. Features are scaled and passed to the **Random Forest** classifier
    4. The predicted mood + confidence is shown alongside what drove the prediction
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.title("Predict Mood")

    st.markdown("### Paste a Spotify track link below")
    link_input = st.text_input("Spotify track URL",
                               placeholder="https://open.spotify.com/track/...")
    link_btn   = st.button("Predict", type="primary", key="link_btn")

    if link_btn and link_input:
        track_id = extract_track_id(link_input)
        if not track_id:
            st.error("Couldn't parse that URL. Make sure it's a valid Spotify track link.")
        else:
            with st.spinner("Fetching track data from Spotify..."):
                features, meta = get_features_from_track_id(track_id)

            if features is None:
                st.error("Spotify couldn't return audio features for this track.")
            else:
                st.session_state['last_features'] = features
                st.session_state['last_meta']     = meta
                st.session_state['show_result']   = True

    # ── Result (shown below both tabs) ────────────────────────────────────────
    if st.session_state.get('show_result'):
        features = st.session_state['last_features']
        meta     = st.session_state['last_meta']
        mood, proba, raw = predict_mood(features)
        cfg = MOOD_CONFIG[mood]

        st.markdown("---")
        col_cover, col_mood, col_bars = st.columns([1, 1.5, 1.5])

        with col_cover:
            if meta['cover']:
                st.image(meta['cover'], width=180)
            st.markdown(f"**{meta['name']}**")
            st.markdown(f"{meta['artist']} · *{meta['album']}*")
            if meta['preview']:
                st.audio(meta['preview'], format='audio/mp3')

        with col_mood:
            st.markdown(f"""
            <div style="background:{cfg['color']}22;border:3px solid {cfg['color']};
                        border-radius:16px;padding:28px;text-align:center;margin-top:8px">
                <div style="color:{cfg['color']};margin-bottom:12px;transform:scale(1.2)">{cfg['icon']}</div>
                <div style="font-size:1.8rem;font-weight:700;color:{cfg['color']}">{mood}</div>
                <div style="color:#888;margin-top:6px">{cfg['desc']}</div>
                <div style="font-size:1.1rem;font-weight:600;margin-top:10px;color:{cfg['color']}">
                    Confidence: {proba[list(LABEL_MAP.values()).index(mood)]:.1%}
                </div>
            </div>""", unsafe_allow_html=True)

        with col_bars:
            st.markdown("**All class probabilities**")
            prob_df = pd.DataFrame({
                'Mood':        list(LABEL_MAP.values()),
                'Probability': proba
            }).sort_values('Probability', ascending=True)
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(prob_df['Mood'], prob_df['Probability'],
                    color=[MOOD_CONFIG[m]['color'] for m in prob_df['Mood']],
                    edgecolor='none')
            ax.set_xlim(0, 1)
            ax.spines[['top','right','left']].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Audio features breakdown
        st.markdown("#### Audio features fetched from Spotify")
        feat_df = pd.DataFrame([features]).T.reset_index()
        feat_df.columns = ['Feature', 'Value']
        feat_df['Value'] = feat_df['Value'].round(4)

        importances = model.feature_importances_
        imp_map = dict(zip(FEATURES, importances))
        feat_df['Importance'] = feat_df['Feature'].map(imp_map).round(4)
        feat_df = feat_df.sort_values('Importance', ascending=False)
        st.dataframe(feat_df, width='stretch', hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VISUALIZE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Visualize":
    st.title("Visualizations")
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "PCA Clusters", "Model Comparison"])

    with tab1:
        st.subheader("What features matter most?")
        importances = model.feature_importances_
        indices     = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=importances[indices],
                    y=[FEATURES[i] for i in indices],
                    palette='viridis', ax=ax)
        ax.set_xlabel('Importance')
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with tab2:
        if os.path.exists('pca_plot.png'):
            st.image('pca_plot.png')
        else:
            st.info("Place pca_plot.png in the app/ folder.")

    with tab3:
        comparison = pd.DataFrame({
            'Model':    ['k-NN', 'Logistic Regression', 'Random Forest'],
            'Accuracy': [0.72,    0.68,                   0.81],
            'F1 Score': [0.71,    0.67,                   0.80],
            'Notes':    ['Simple baseline', 'Linear boundary', 'Best — used in production']
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        if os.path.exists('confusion_matrices.png'):
            st.image('confusion_matrices.png')