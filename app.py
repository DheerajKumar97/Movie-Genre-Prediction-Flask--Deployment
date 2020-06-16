# Importing essential libraries
from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
mnb = pickle.load(open('Naive_Bayes_model.pkl','rb'))
countVect = pickle.load(open('countVect_Genre.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        text = request.form['text']
        data = [text]
        vect = countVect.transform(data).toarray()
        my_prediction = mnb.predict(vect)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
    