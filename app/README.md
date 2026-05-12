# Honest CAN-IDS Simulator — Live Demo

Streamlit companion app for the makale3 paper. Generates a synthetic CAN
trace on demand, feeds it through a reference random-forest detector, and
visualises the trace, the detector confidence, the calibration curve, and
the observation-asymmetry channel.

## Run locally

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

The app pulls its simulator backend from `code/utils/sim.py`; no extra
configuration is needed.

## Deploy on Streamlit Cloud (free tier)

1. Push the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   the same GitHub account.
3. **New app** → pick this repo and the branch you want to track.
4. **Main file path**: `app/streamlit_app.py`
5. **Advanced settings → requirements file**: `app/requirements.txt`
6. Deploy. The first build takes 2-3 minutes; subsequent builds are
   incremental.

The app's reference detector is trained once at first load (a few seconds
of cold-start cost) and cached for the life of the session via
`@st.cache_resource`.
