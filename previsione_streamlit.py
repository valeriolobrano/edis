import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import datetime
import requests
import altair as alt

mesi_it = {
    1: "Gennaio", 2: "Febbraio", 3: "Marzo", 4: "Aprile",
    5: "Maggio", 6: "Giugno", 7: "Luglio", 8: "Agosto",
    9: "Settembre", 10: "Ottobre", 11: "Novembre", 12: "Dicembre"
}

data_domani = datetime.date.today() + datetime.timedelta(days=1)
giorno = data_domani.day
mese = mesi_it[data_domani.month]
anno = data_domani.year

st.title(f"📈 Previsione Temperatura Media – CP 4 Mandamenti per il giorno {giorno} {mese} {anno}")

# --- Titolo e introduzione ---
st.set_page_config(layout="wide")
# Calcola la data di domani in formato leggibile
# Calcola la data di domani e formatta in italiano
data_domani = datetime.datetime.now(datetime.timezone.utc).date() + datetime.timedelta(days=1)
data_formattata = data_domani.strftime("%d %B %Y")  # Esempio: 27 giugno 2025

# Titolo dinamico con data
st.markdown("Previsione automatica della temperatura media interna per CP 4 Mandamenti usando dati Open-Meteo e modello CatBoost ottimizzato. Indice di correlazione **R² = 0.98**")

# --- Parametri ---
potenza_massima = st.slider("🌞 Imposta il massimo della potenza prevista (MW)", min_value=10, max_value=45, value=30)

# --- Data corrente e previsione per domani ---
odierna = datetime.datetime.now(datetime.timezone.utc).date()
data_domani = odierna + datetime.timedelta(days=1)

# --- Coordinate Palermo ---
latitude = 38.1157
longitude = 13.3615

# --- Download previsioni Open-Meteo ---
@st.cache_data(show_spinner=True)
def get_openmeteo_forecast():
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,shortwave_radiation&timezone=auto&start_date={data_domani}&end_date={data_domani}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data['hourly'])
    df['hour'] = pd.to_datetime(df['time'])
    df.rename(columns={"shortwave_radiation": "ghi", "temperature_2m": "Tair2m"}, inplace=True)
    return df[['hour', 'ghi', 'Tair2m']]

# --- Caricamento modello ---
@st.cache_resource(show_spinner=False)
def load_model():
    model = CatBoostRegressor()
    model.load_model("CatBoost-ADV.cbm")
    return model

# --- Prepara dataframe con feature ---
def costruisci_feature(df_meteo, potenza_massima):
    df_input = df_meteo.copy()
    df_input['ghi_lag1'] = df_input['ghi'].shift(1).fillna(method='bfill')
    df_input['ghi_lag2'] = df_input['ghi'].shift(2).fillna(method='bfill')

    df_input['hour_of_day'] = df_input['hour'].dt.hour
    df_input['month_of_year'] = df_input['hour'].dt.month
    df_input['sin_hour'] = np.sin(2 * np.pi * df_input['hour_of_day'] / 24)
    df_input['cos_hour'] = np.cos(2 * np.pi * df_input['hour_of_day'] / 24)
    df_input['sin_month'] = np.sin(2 * np.pi * df_input['month_of_year'] / 12)
    df_input['cos_month'] = np.cos(2 * np.pi * df_input['month_of_year'] / 12)
    df_input['is_weekend'] = df_input['hour'].dt.weekday >= 5
    df_input['is_weekend'] = df_input['is_weekend'].astype(int)

    # Giorno della settimana (one-hot)
    df_input['day_of_week'] = df_input['hour'].dt.day_name()
    df_input = pd.get_dummies(df_input, columns=['day_of_week'], drop_first=True)

    # Aggiungi colonne mancanti per compatibilità col modello
    day_of_week_features = [
        'day_of_week_Monday',
        'day_of_week_Saturday',
        'day_of_week_Sunday',
        'day_of_week_Thursday',
        'day_of_week_Tuesday',
        'day_of_week_Wednesday'
    ]
    for col in day_of_week_features:
        if col not in df_input.columns:
            df_input[col] = 0

    # Andamento potenza pseudosinusoidale
    def andamento_potenza(ore, potenza_massima):
        secondi_giorno = ore.dt.hour * 3600 + ore.dt.minute * 60
        secondi_min = 6 * 3600 + 30 * 60
        secondi_max = 14 * 3600
        ampiezza = (potenza_massima - 10) / 2
        offset = 10 + ampiezza
        normalizzati = (secondi_giorno - secondi_min) / (secondi_max - secondi_min) * np.pi
        normalizzati = np.clip(normalizzati, 0, np.pi)
        return offset + ampiezza * np.sin(normalizzati - np.pi/2)

    df_input['potenza'] = andamento_potenza(df_input['hour'], potenza_massima)
    df_input['potenza_lag1'] = df_input['potenza'].shift(1).fillna(method='bfill')
    df_input['potenza_lag2'] = df_input['potenza'].shift(2).fillna(method='bfill')

    return df_input

# --- MAIN ---
try:
    df_meteo = get_openmeteo_forecast()
    df_input = costruisci_feature(df_meteo, potenza_massima)

    # Feature usate durante l’addestramento
    feature_cols = [
        'ghi', 'ghi_lag1', 'ghi_lag2', 'Tair2m', 'potenza',
        'potenza_lag1', 'potenza_lag2',
        'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
        'is_weekend',
        'day_of_week_Monday',
        'day_of_week_Saturday',
        'day_of_week_Sunday',
        'day_of_week_Thursday',
        'day_of_week_Tuesday',
        'day_of_week_Wednesday'
    ]

    model = load_model()
    y_pred = model.predict(df_input[feature_cols])

    df_input['Previsione Temperatura Media'] = y_pred
    df_input['Ora'] = df_input['hour'].dt.strftime('%H:%M')

     st.success("✅ Previsione completata con successo!")

    # --- Intervallo di confidenza fittizio (±0.6 °C) ---
    df_input['Conf_Lower'] = df_input['Previsione Temperatura Media'] - 0.6
    df_input['Conf_Upper'] = df_input['Previsione Temperatura Media'] + 0.6

    base = alt.Chart(df_input).encode(x=alt.X('Ora', title='Orario'))

    area = base.mark_area(opacity=0.3, color='lightblue').encode(
        y='Conf_Lower',
        y2='Conf_Upper'
    )

    line = base.mark_line(color='blue').encode(
        y=alt.Y('Previsione Temperatura Media', title='Temperatura [°C]', scale=alt.Scale(domain=[30, 40])),
        tooltip=['Ora', 'Previsione Temperatura Media']
    )

    chart = (area + line).properties(
        title='📈 Previsione Temperatura Media della CP Rossa 4 Mandamenti',
        width=800,
        height=300
    )

    st.altair_chart(chart, use_container_width=True)
    

except Exception as e:
    st.error(f"Errore durante la previsione: {e}")
