from flask import Flask, render_template, request
import pickle
import numpy as np
import sklearn
import sklearn.preprocessing import train_test_split
import pandas as pd

app = Flask(__name__)


model = pickle.load(open("linear_model.pkl", "rb"))

df = pd.read_csv("insurence.csv")

X = df.drop("charges", axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=322)

# function url -> front part / end part facebook.com


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def prediction():

    if request.method == "POST":
        age = request.form['age']
        bmi = request.form['bmi']
        children = request.form['children']
        sex = request.form['sex']
        smoker = request.form['smoker']
        region = request.form['region']

        predict = model.predict([[age, bmi, children, region, sex, smoker]])
        accuracy = model.score(x_test)

        return render_template('index.html', predict="You premium amount {}".format(predict))
    else:
        return render_template("index.html", predict="Not Predicted")


if __name__ == '__main__':
    app.run(debug=True)
