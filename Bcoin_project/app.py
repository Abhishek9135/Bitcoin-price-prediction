from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "bitcoin_lstm_model.h5"
SCALER_PATH = BASE_DIR / "scaler.pkl"
SEQUENCE_LENGTH = 60


st.set_page_config(page_title="Bitcoin Predictor", layout="centered")


@st.cache_resource(show_spinner=False)
def load_assets():
    try:
        from tensorflow.keras.models import Sequential, load_model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH.name}")

    if not SCALER_PATH.exists():
        raise RuntimeError(f"Scaler file not found: {SCALER_PATH.name}")

    def strip_config_key(value, key_to_remove):
        if isinstance(value, dict):
            return {
                key: strip_config_key(child, key_to_remove)
                for key, child in value.items()
                if key != key_to_remove
            }
        if isinstance(value, list):
            return [strip_config_key(item, key_to_remove) for item in value]
        return value

    # Some older H5 model files include `quantization_config=None` on Dense layers,
    # which Keras 3 rejects during deserialization. Fallback to a sanitized config.
    def load_saved_model():
        try:
            return load_model(MODEL_PATH, compile=False)
        except TypeError as exc:
            if "quantization_config" not in str(exc):
                raise

            try:
                import h5py
            except ModuleNotFoundError as h5py_exc:
                raise RuntimeError(
                    "h5py is required to load the saved model compatibility fallback."
                ) from h5py_exc

            with h5py.File(MODEL_PATH, "r") as model_file:
                raw_config = json.loads(model_file.attrs["model_config"])

            if raw_config.get("class_name") != "Sequential":
                raise RuntimeError(
                    "The saved model uses an unsupported architecture for the compatibility loader."
                )

            clean_config = strip_config_key(raw_config, "quantization_config")
            model = Sequential.from_config(clean_config["config"])
            model.load_weights(MODEL_PATH)
            return model

    try:
        model = load_saved_model()
    except Exception as exc:
        raise RuntimeError(f"Unable to load {MODEL_PATH.name}: {exc}") from exc

    try:
        with SCALER_PATH.open("rb") as scaler_file:
            scaler = pickle.load(scaler_file)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "scikit-learn is required to load scaler.pkl. Run `pip install -r requirements.txt` first."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Unable to load {SCALER_PATH.name}: {exc}") from exc

    return model, scaler


def get_live_data():
    try:
        import pandas as pd
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pandas and requests are required. Run `pip install -r requirements.txt` first."
        ) from exc

    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "1"}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Unable to fetch live Bitcoin data: {exc}") from exc

    payload = response.json()
    prices = payload.get("prices", [])
    if not prices:
        raise RuntimeError("The API did not return any price data.")

    df = pd.DataFrame(prices, columns=["timestamp_ms", "price_usd"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
    return df


def preprocess(df, scaler):
    if len(df) < SEQUENCE_LENGTH:
        raise RuntimeError(
            f"Not enough data points to predict. Need at least {SEQUENCE_LENGTH}, got {len(df)}."
        )

    last_prices = df[["price_usd"]].tail(SEQUENCE_LENGTH).astype(float).copy()
    if hasattr(scaler, "feature_names_in_") and len(scaler.feature_names_in_) == 1:
        last_prices.columns = [scaler.feature_names_in_[0]]

    scaled_prices = scaler.transform(last_prices)
    return np.expand_dims(scaled_prices, axis=0)


def predict_price(model, scaler, features):
    prediction = model.predict(features, verbose=0)
    return float(scaler.inverse_transform(prediction)[0][0])


def main():
    st.title("Bitcoin Price Predictor")
    st.write("Fetch live Bitcoin prices and generate a one-step forecast with the saved LSTM model.")

    try:
        model, scaler = load_assets()
    except RuntimeError as error:
        st.error(str(error))
        st.stop()

    if st.button("Predict now", type="primary"):
        with st.spinner("Fetching live market data and running the model..."):
            try:
                df = get_live_data()
                features = preprocess(df, scaler)
                predicted_price = predict_price(model, scaler, features)
            except RuntimeError as error:
                st.error(str(error))
                return

        latest_price = float(df["price_usd"].iloc[-1])
        st.metric(
            label="Predicted BTC price",
            value=f"${predicted_price:,.2f}",
            delta=f"${predicted_price - latest_price:,.2f}",
        )
        st.caption(f"Latest observed BTC price: ${latest_price:,.2f}")

        chart_data = df.set_index("timestamp")[["price_usd"]].rename(
            columns={"price_usd": "BTC price (USD)"}
        )
        st.line_chart(chart_data)


if __name__ == "__main__":
    main()
