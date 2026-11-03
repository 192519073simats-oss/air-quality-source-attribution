# AI-Powered Air Quality Monitoring & Source Attribution

Hackathon prototype that fuses satellite-related and ground/weather/activity features to:
- estimate PM2.5 concentration (Random Forest regression)
- classify likely dominant pollution source (Random Forest classification)
- display trends, model metrics, and feature importance in a Streamlit dashboard

## Important data note
`data/synthetic_air_quality.csv` is **synthetic demonstration data**, not real measured air quality or actual satellite retrievals. `AOD` is represented as a satellite-derived feature proxy; activity indices are illustrative. Do not use this prototype for public-health decisions without replacing the sample with validated data.

## Repository structure
```
air_quality_source_attribution/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── synthetic_air_quality.csv
```

## Run locally
Python 3.9+ recommended.

```bash
git clone https://github.com/YOUR-USERNAME/air-quality-source-attribution.git
cd air-quality-source-attribution
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
The app opens in your browser (usually http://localhost:8501).

## Dataset columns
| Column | Description |
|---|---|
| timestamp | Synthetic hourly timestamp |
| AOD | Aerosol Optical Depth proxy |
| temperature_c | Temperature |
| humidity_pct | Relative humidity |
| wind_speed_m_s | Wind speed |
| traffic_index | Illustrative traffic activity index (0–100) |
| industrial_activity_index | Illustrative industry activity index (0–100) |
| green_cover_pct | Illustrative green cover percentage |
| hour, weekday | Time features |
| PM2_5 | Simulated target concentration, µg/m³ |
| source_label | Simulated dominant source label |

## Proposed real-data upgrade
1. **Ground data:** OpenAQ API or CPCB station observations (respect access terms and QA flags).
2. **Satellite:** NASA MAIAC/MODIS or Sentinel-5P/TROPOMI products; select AOD / aerosol and atmospheric variables with temporal-spatial matching.
3. **Weather:** ERA5 or station meteorology.
4. **Spatial matching:** Join satellite pixel + ground station + weather by coordinates and time window; handle missing/cloud-screened observations.
5. **Train/evaluate:** time-based split (avoid random leakage), compare Random Forest, XGBoost/LightGBM; report MAE, RMSE, R² and calibration.
6. **Source attribution:** combine model interpretation (SHAP) with wind direction, back trajectories, emission inventories, and geospatial proximity. A classifier alone cannot establish causal source attribution.

## Limitations
- No live satellite feed or sensor connection in this starter.
- Synthetic labels can make model scores look better than real-world performance.
- “Likely source” is a hypothesis; source apportionment requires validated measurements and additional evidence.
