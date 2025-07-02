'''
versione 02/07/2025
'''
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import datetime
import requests
import altair as alt

# --- Configurazione pagina ---
st.set_page_config(layout="wide")
# Versione script in alto a destra
st.markdown(
    "<span style='float:right;font-size:12px;color:gray;'>Versione 02/07/2025</span>",
    unsafe_allow_html=True
)

# --- Date ---
mesi_it = {
    1: "Gennaio", 2: "Febbraio", 3: "Marzo", 4: "Aprile",
    5: "Maggio", 6: "Giugno", 7: "Luglio", 8: "Agosto",
    9: "Settembre", 10: "Ottobre", 11: "Novembre", 12: "Dicembre"
}
oggi = datetime.datetime.now(datetime.timezone.utc).date()
data_domani = oggi + datetime.timedelta(days=1)
giorno = data_domani.day
mese = mesi_it[data_domani.month]
anno = data_domani.year

# Titolo principale
st.title(f"📈 Previsione Temperatura Media – CP 4 Mandamenti per il giorno {giorno} {mese} {anno}")
st.markdown(
    "Previsione automatica della temperatura media interna per CP 4 Mandamenti usando dati Open-Meteo e modello CatBoost ottimizzato. Indice di correlazione **R² = 0.98**"
)

# --- Parametri di potenza ---
col1, col2 = st.columns(2)
with col1:
    potenza_minima = st.slider(
        "⚡ Imposta il minimo della potenza prevista (MW)",
        min_value=0, max_value=30, value=10
    )
with col2:
    potenza_massima = st.slider(
        "⚡ Imposta il massimo della potenza prevista (MW)",
        min_value=potenza_minima, max_value=45, value=30
    )

# --- Coordinate Palermo ---
latitude = 38.1157
longitude = 13.3615

# --- Download previsioni Open-Meteo ---
@st.cache_data(show_spinner=True)
def get_openmeteo_forecast():
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}"
        f"&longitude={longitude}&hourly=temperature_2m,shortwave_radiation"
        f"&timezone=auto&start_date={data_domani}&end_date={data_domani}"
    )
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data['hourly'])
    df['hour'] = pd.to_datetime(df['time'])
    df.rename(columns={
        "shortwave_radiation": "ghi",
        "temperature_2m": "Tair2m"
    }, inplace=True)
    return df[['hour', 'ghi', 'Tair2m']]

# --- Caricamento modello CatBoost ---
@st.cache_resource(show_spinner=False)
def load_model():
    model = CatBoostRegressor()
    model.load_model("CatBoost-ADV.cbm")
    return model

# --- Costruzione delle feature ---
def costruisci_feature(df_meteo, potenza_min, potenza_max):
    df_input = df_meteo.copy()
    # Lag ghi
    df_input['ghi_lag1'] = df_input['ghi'].shift(1).bfill()
    df_input['ghi_lag2'] = df_input['ghi'].shift(2).bfill()

    # Time features
    df_input['hour_of_day'] = df_input['hour'].dt.hour
    df_input['month_of_year'] = df_input['hour'].dt.month
    df_input['sin_hour'] = np.sin(2 * np.pi * df_input['hour_of_day'] / 24)
    df_input['cos_hour'] = np.cos(2 * np.pi * df_input['hour_of_day'] / 24)
    df_input['sin_month'] = np.sin(2 * np.pi * df_input['month_of_year'] / 12)
    df_input['cos_month'] = np.cos(2 * np.pi * df_input['month_of_year'] / 12)
    df_input['is_weekend'] = (df_input['hour'].dt.weekday >= 5).astype(int)

    # One-hot per giorno settimana
    df_input['day_of_week'] = df_input['hour'].dt.day_name()
    df_input = pd.get_dummies(df_input, columns=['day_of_week'], drop_first=True)
    for col in [
        'day_of_week_Monday', 'day_of_week_Tuesday', 'day_of_week_Wednesday',
        'day_of_week_Thursday', 'day_of_week_Saturday', 'day_of_week_Sunday'
    ]:
        if col not in df_input.columns:
            df_input[col] = 0

    # Andamento potenza pseudosinusoidale
    def andamento_potenza(ore, p_min, p_max):
        secondi = ore.dt.hour * 3600 + ore.dt.minute * 60
        sec_min = 6 * 3600 + 30 * 60
        sec_max = 14 * 3600
        amp = (p_max - p_min) / 2
        off = p_min + amp
        norm = (secondi - sec_min) / (sec_max - sec_min) * np.pi
        norm = np.clip(norm, 0, np.pi)
        return off + amp * np.sin(norm - np.pi/2)

    df_input['potenza'] = andamento_potenza(df_input['hour'], potenza_min, potenza_max)
    df_input['potenza_lag1'] = df_input['potenza'].shift(1).bfill()
    df_input['potenza_lag2'] = df_input['potenza'].shift(2).bfill()

    return df_input

# --- Main ---
try:
    df_meteo = get_openmeteo_forecast()
    df_input = costruisci_feature(df_meteo, potenza_minima, potenza_massima)

    # Previsione
    feature_cols = [
        'ghi', 'ghi_lag1', 'ghi_lag2', 'Tair2m', 'potenza',
        'potenza_lag1', 'potenza_lag2',
        'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
        'is_weekend',
        'day_of_week_Monday', 'day_of_week_Tuesday', 'day_of_week_Wednesday',
        'day_of_week_Thursday', 'day_of_week_Saturday', 'day_of_week_Sunday'
    ]
    model = load_model()
    y_pred = model.predict(df_input[feature_cols])
    df_input['Previsione Temperatura Media'] = y_pred
    df_input['Ora'] = df_input['hour'].dt.strftime('%H:%M')

    st.success("✅ Previsione completata con successo!")

    # --- Calcoli statistici ---
    max_pred = df_input['Previsione Temperatura Media'].max()
    time_max = df_input.loc[df_input['Previsione Temperatura Media'].idxmax(), 'Ora']
    avg_pred = df_input['Previsione Temperatura Media'].mean()

    st.markdown(f"**🔥 Massimo previsto:** **{max_pred:.1f}°C ±1°C** alle **{time_max}**")
    st.markdown(f"**📊 Media giornaliera:** **{avg_pred:.1f}°C**")

    # --- Calcola limiti asse Y ---
    min_val = min(df_input['Tair2m'].min(), df_input['Previsione Temperatura Media'].min())
    max_val = max(df_input['Tair2m'].max(), df_input['Previsione Temperatura Media'].max())
    margin = 2  # gradi di margine
    y_min = min_val - margin
    y_max = max_val + margin

    # --- Grafico multi-serie ---
    df_plot = df_input[['Ora', 'Tair2m', 'Previsione Temperatura Media']]
    df_plot = df_plot.melt(id_vars='Ora', var_name='Serie', value_name='Valore')
    chart = alt.Chart(df_plot).mark_line().encode(
        x=alt.X('Ora:N', title='Orario'),
        y=alt.Y('Valore:Q', title='Temperatura [°C]',
                scale=alt.Scale(domain=[y_min, y_max])),
        color=alt.Color(
            'Serie:N',
            scale=alt.Scale(
                domain=['Tair2m', 'Previsione Temperatura Media'],
                range=['#1f77b4', '#ff7f0e']
            ),
            legend=alt.Legend(title="Serie")
        ),
        tooltip=['Ora', 'Serie', 'Valore']
    ).properties(
        title={'text':'📈 Previsione Temperatura Interna vs Temperatura aria Esterna','anchor':'middle'},
        width=800, height=450
    )
    st.altair_chart(chart, use_container_width=True)

except Exception as e:
    st.error(f"Errore durante la previsione: {e}")
