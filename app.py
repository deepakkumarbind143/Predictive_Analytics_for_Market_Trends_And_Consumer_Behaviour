from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("model/model.h5")


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error_message = None

    if request.method == 'POST':
        ticker = request.form['ticker'].strip().upper()

        try:
            data = yf.download(ticker, start="2018-01-01", end="2023-01-01")

            if data.empty or 'Close' not in data:
                raise ValueError("No data found for this ticker.")

            data = data[['Close']]
            if len(data) < 61:
                raise ValueError("Not enough data to make a prediction.")

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)

            def create_dataset(dataset, time_step=60):
                X = []
                for i in range(len(dataset) - time_step - 1):
                    X.append(dataset[i:(i + time_step), 0])
                return np.array(X)

            dataset = create_dataset(scaled_data)
            if len(dataset) == 0:
                raise ValueError("Not enough historical data.")

            X_input = dataset[-1].reshape(1, 60, 1)
            prediction = model.predict(X_input)
            prediction = scaler.inverse_transform(prediction)[0][0]
            prediction = round(prediction, 2)

        except Exception:
            error_message = "📢 This model is still getting trained. Sorry for the inconvenience."

    return render_template("index.html", prediction=prediction, error=error_message)


if __name__ == '__main__':
    app.run(debug=True)

