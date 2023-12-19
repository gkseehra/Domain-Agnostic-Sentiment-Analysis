from flask import Flask, request, render_template
import pandas as pd
import joblib
from transformers import pipeline


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # model = joblib.load('best_rf_tf_idf.pkl')
    model = joblib.load('best_svm_tf_idf_classifier.pkl')

    loaded_vectorizer = joblib.load('tfidfVectorizer.pkl')

    sa_input_data = request.form.get('sa-data-input-sub')
    df = pd.DataFrame({'Reviews': [sa_input_data]})

    input = loaded_vectorizer.transform(df['Reviews'])
    X_train_tfidf_v_df = pd.DataFrame(input.toarray())
    X_train_tfidf_v_df.columns = loaded_vectorizer.get_feature_names_out()

    prediction = model.predict(X_train_tfidf_v_df).tolist()[0]

    # Using transformers
    # sentiment_pipeline = pipeline("sentiment-analysis")
    # data = [sa_input_data]
    # ans = sentiment_pipeline(data)[0]['label']
    # print(">>>>>>>>>>>ans", ans)
    # print(">>>>>>>>>>>prediction", prediction)

    res = "NEGATIVE"
    if prediction == "__label__2":
        res = "POSITIVE"

    return render_template('result.html', output=res)


if __name__ == '__main__':
    app.run(debug=True)
