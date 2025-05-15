from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoders
with open("model_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)
    model = bundle["model"]
    encoders = bundle["encoders"]
    n_trees = bundle["n_trees"]

brand_model_map = {
    "Audemars Piguet": ["Royal Oak", "Code 11.59", "Jules Audemars"],
    "Patek Philippe": ["Nautilus", "Calatrava", "Aquanaut", "Annual Calendar"],
    "Tag Heuer": ["Formula 1", "Carrera", "Aquaracer"],
    "Grand Seiko": ["Grand Seiko"],
    "Casio": ["G-Shock", "Edifice", "F91W", "Pro Trek", "Databank"],
    "Hublot": ["Big Bang", "Spirit of Big Bang", "Classic Fusion"],
    "Rolex": ["Submariner", "Datejust", "Day-Date", "Oyster Perpetual"],
    "Seiko": ["Presage", "5 Sports", "Astron"],
    "Longines": ["HydroConquest", "Conquest"],
    "Cartier": ["Tank", "Drive", "Santos", "Must de Cartier"],
    "Tudor": ["Black Bay", "Heritage Collection"],
    "Omega": ["Speedmaster", "Seamaster", "De Ville", "Constellation", "Railmaster"],
    "Citizen": ["Eco-Drive", "Promaster", "Chandler", "Skyhawk", "Navihawk"]
}

def predict_with_uncertainty(model, X_input, n_trees):
    preds = np.array([
        model.predict(X_input, iteration_range=(0, i + 1)) for i in range(n_trees)
    ])
    mean_pred = preds[-1]
    std_pred = np.std(preds, axis=0)
    return mean_pred[0], std_pred[0]

@app.route("/")
def home():
    brand_list = encoders["Brand"].classes_
    model_list = encoders["Model"].classes_
    return render_template("home.html", brand_model_map=brand_model_map)


@app.route("/predict", methods=["POST"])
def predict_form():
    try:
        brand = request.form["brand"]
        model_name = request.form["model"]
        condition_label = request.form["condition"]

        condition_map = {"New": 1.0, "Used": 0.5, "Damaged": 0.1}
        condition = condition_map.get(condition_label, 0.0)

        # Validate inputs against encoder classes
        if brand not in encoders["Brand"].classes_:
            return f"Sorry, we don't support the brand '{brand}' yet."

        if model_name not in encoders["Model"].classes_:
            return f"Sorry, the model '{model_name}' is not recognized. Please choose from supported models."

        brand_encoded = encoders["Brand"].transform([brand])[0]
        model_encoded = encoders["Model"].transform([model_name])[0]

        X_input = np.array([[brand_encoded, model_encoded, condition]])
        price, uncertainty = predict_with_uncertainty(model, X_input, n_trees)

        return render_template("after.html",
                               predicted_price=round(price, 2),
                               uncertainty=round(uncertainty, 2))
    except Exception as e:
        return f"Unexpected error: {e}"

if __name__ == "__main__":
    app.run(debug=True)