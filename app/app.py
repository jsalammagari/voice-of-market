from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('enhanced_sentiment_analysis_model.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        tweet = request.form['tweet']
        # Here you would typically preprocess the tweet text
        # Since we don't have the preprocessing steps, we'll use the raw text
        prediction = model.predict([tweet])
        print(prediction)
        return render_template('index.html', sentiment=prediction)
    return render_template('index.html', sentiment=None)

if __name__ == '__main__':
    app.run(debug=True)
