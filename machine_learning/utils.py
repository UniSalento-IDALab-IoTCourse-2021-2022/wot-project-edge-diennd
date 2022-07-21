import joblib


def predict_hf(data):

    #trained_model = joblib.load("model/model.pkl") #this model version to be used in Ubuntu Desktop
    trained_model = joblib.load("model/LogisticRegression_model.pkl")  #this version to be used in the Pi
    predictions = trained_model.predict(data)

    return predictions
