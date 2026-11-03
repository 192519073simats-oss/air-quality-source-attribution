import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from pathlib import Path

st.set_page_config(page_title="Air Quality AI", page_icon="🌍", layout="wide")
st.title("🌍 AI Air Quality Monitoring & Source Attribution")
st.caption("Hackathon prototype • ML prediction + exploratory source attribution")

DATA_PATH = Path(__file__).parent / "data" / "synthetic_air_quality.csv"
df = pd.read_csv(DATA_PATH)
features = ["AOD", "temperature_c", "humidity_pct", "wind_speed_m_s",
            "traffic_index", "industrial_activity_index", "green_cover_pct", "hour", "weekday"]
X = df[features]
y_pm = df["PM2_5"]
y_src = df["source_label"]

X_train, X_test, yp_train, yp_test = train_test_split(X, y_pm, test_size=0.2, random_state=42)
reg = RandomForestRegressor(n_estimators=180, random_state=42)
reg.fit(X_train, yp_train)
pred = reg.predict(X_test)
Xc_train, Xc_test, ys_train, ys_test = train_test_split(X, y_src, test_size=0.2, random_state=42, stratify=y_src)
clf = RandomForestClassifier(n_estimators=180, random_state=42, class_weight="balanced")
clf.fit(Xc_train, ys_train)
src_pred = clf.predict(Xc_test)

c1,c2,c3 = st.columns(3)
c1.metric("Model R² (PM2.5)", f"{r2_score(yp_test,pred):.3f}")
c2.metric("PM2.5 MAE", f"{mean_absolute_error(yp_test,pred):.1f} µg/m³")
c3.metric("Source classification accuracy", f"{accuracy_score(ys_test,src_pred):.1%}")

st.warning("Dataset bundled with this prototype is SYNTHETIC and intended for demonstration. Replace it with calibrated ground sensors and satellite data before operational use.")

st.subheader("Explore sample monitoring data")
col1,col2=st.columns(2)
with col1:
    st.line_chart(df.set_index("timestamp")["PM2_5"].rolling(24).mean(), height=260)
    st.caption("24-hour rolling PM2.5 (synthetic)")
with col2:
    st.bar_chart(df["source_label"].value_counts(), height=260)
    st.caption("Sample source-label distribution")

st.subheader("Estimate air quality for a location/time")
with st.form("predict"):
    a,b,c=st.columns(3)
    AOD=a.slider("Satellite AOD proxy",0.0,1.5,0.35,0.01)
    temp=b.slider("Temperature (°C)",10.0,50.0,30.0,0.5)
    hum=c.slider("Humidity (%)",0.0,100.0,55.0,1.0)
    d,e,f=st.columns(3)
    wind=d.slider("Wind speed (m/s)",0.0,15.0,2.5,0.1)
    traffic=e.slider("Traffic index",0.0,100.0,45.0,1.0)
    industry=f.slider("Industrial activity index",0.0,100.0,35.0,1.0)
    g,h,i=st.columns(3)
    green=g.slider("Green cover (%)",0.0,100.0,35.0,1.0)
    hour=h.select_slider("Hour of day",options=list(range(24)),value=14)
    weekday=i.select_slider("Day of week (0=Mon)",options=list(range(7)),value=2)
    submitted=st.form_submit_button("Predict air quality")
if submitted:
    sample=pd.DataFrame([[AOD,temp,hum,wind,traffic,industry,green,hour,weekday]],columns=features)
    pm=float(reg.predict(sample)[0])
    source=clf.predict(sample)[0]
    if pm<=12: category="Good"
    elif pm<=35: category="Moderate"
    elif pm<=55: category="Poor"
    else: category="Severe"
    st.metric("Predicted PM2.5",f"{pm:.1f} µg/m³",category)
    st.info(f"Likely dominant source (model estimate): **{source}**")
    st.caption("Source attribution is a statistical estimate, not proof of causation. Validate with domain experts and real emissions inventories.")

st.subheader("What drives the model?")
imp=pd.Series(reg.feature_importances_,index=features).sort_values()
fig,ax=plt.subplots(figsize=(8,4))
imp.plot(kind="barh",ax=ax)
ax.set_xlabel("Random Forest feature importance")
st.pyplot(fig)
st.download_button("Download bundled dataset",df.to_csv(index=False),file_name="synthetic_air_quality.csv",mime="text/csv")
