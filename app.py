from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model_SVC = pickle.load(open('model_SVC.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_values = list(request.form.values())

        city_map = {
            "Karachi": 0, "Islamabad": 1, "Rawalpindi": 2, "Lahore": 3,
            "Gujranwala": 4, "Faislabad": 5, "Multan": 6, "Hyderabad": 7
        }
        engine_map = {
            "Single Cylinder": 0, "Inline-4": 1, "V-Twin": 2,
            "Twin Cylinder": 3, "Parallel Twin": 4
        }
        stroke_map = {"2-Stroke": 0, "4-Stroke": 1}

        bike_name_map = {
            0: "Honda CB 150F",
            1: "Yamaha MT-15",
            2: "Kawasaki Ninja 300",
            3: "Suzuki GSX-R600",
            4: "United US 125",
            5: "Super Power SP 70",
            6: "Road Prince 110",
            7: "Metro MR 70"
        }

        # Extract and map form values
        # Adjust the indices if your form order is different
        bike_price = int(form_values[2])
        engine_type = engine_map.get(form_values[3], -1)
        stroke = stroke_map.get(form_values[5], -1)         
        city = city_map.get(form_values[0], -1)      
        kilometer_ran = int(form_values[1])      
        model_year = int(form_values[4])

        # ...existing code...
        # Match the training feature order
        features = [bike_price, engine_type, stroke, city, kilometer_ran, model_year]
        final_input = np.array([features])

        # Get prediction probabilities for all classes
        probabilities = model_SVC.predict_proba(final_input)[0]
        # Get indices of top 4 probabilities
        top_indices = np.argsort(probabilities)[-4:][::-1]

        # Map indices to bike names and probabilities
        top_bikes = [
            f"{bike_name_map.get(idx, 'Unknown Bike')} ({probabilities[idx]*100:.2f}%)"
            for idx in top_indices
        ]
        
        prediction_text = "" + "\n" + "\n".join(top_bikes)
        return render_template('index.html', prediction_text=prediction_text)
# ...existing code...
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
