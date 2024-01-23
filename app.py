import pickle

import nltk
from flask import Flask, render_template, request
from nltk import FreqDist
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
app = Flask(__name__)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum() and word not in stop_words]
    return dict(FreqDist(words))

with open('models/t4.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

@app.route('/', methods=["GET", "POST"])
def hello_world():
    name = request.form.get('name',"non")
    prediction = classifier.prob_classify(preprocess_text(name))
    return render_template("pages/index.html",prediction =prediction)
@app.route('/tester')
def tester():  # put application's code here
    return render_template("pages/test.html")

@app.route('/resultat' , methods=["GET", "POST"])
def resultat():  # put application's code here
    user_text = request.form.get('userText')
    lines = user_text.strip().split('\n')
    pos=0
    neg =0
    non = 0
    for line in lines:
        prediction = classifier.prob_classify(preprocess_text(line))

        if prediction.prob(1) >0.7:
            pos+=1
        elif prediction.prob(0)>0.7:
            neg+=1
        else:
            non+=1
    pos = round(pos * 100 / len(lines), 1)
    neg = round(neg * 100 / len(lines), 1)
    non = round(non * 100 / len(lines), 1)

    i=0
    return render_template("pages/score.html",pos=pos,neg =neg,non = non )

@app.route('/upload')
def upload_file():  # put application's code here
    return render_template("pages/apload.html")

@app.route('/test_file', methods=['POST' ,"GET"])
def test_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename[-4:] == '.txt':

        # Vous pouvez traiter le fichier ici
        data = uploaded_file.read()
        lines = data.decode('utf-8').strip().split("\n")
        # Faites quelque chose avec le contenu du fichier
        print(len(lines))
        pos = 0
        neg = 0
        non = 0
        for line in lines:
            print(line)
            prediction = classifier.prob_classify(preprocess_text(line))
            print("ok2")
            if prediction.prob(1) > 0.7:
                pos += 1
            elif prediction.prob(0) > 0.7:
                neg += 1
            else:
                non += 1
        pos = round(pos * 100 / len(lines),1)
        neg = round(neg * 100 / len(lines),1)
        non = round(non * 100 / len(lines),1)
        return render_template("pages/score.html", pos=pos, neg=neg, non=non)

    return render_template("pages/apload.html")


if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0",port=5000)



@app.route('/about')
def about_page():  # put application's code here
    return render_template("pages/about.html")