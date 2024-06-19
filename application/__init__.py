from flask import Flask, request, Response, json
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import pickle

# scaler loaded
scaler = joblib.load('scaler.gz')

# load the model from disk
filename = 'model.sav'
rf = pickle.load(open(filename, 'rb'))

# flask instance
app = Flask(__name__)

# Create API
@app.route("/api", methods=["GET", "POST"])
def predict():
    # get data from request
    data = request.get_json(force=True)

    Percent_Days_Employed = float(data["Percent_Days_Employed"])
    Supervision_Risk_Score_First = float(data["Supervision_Risk_Score_First"])
    Prior_Arrest_Episodes_PPViolationCharges = float(data["Prior_Arrest_Episodes_PPViolationCharges"])
    Gang_Affiliated = float(data["Gang_Affiliated"])
    Prior_Arrest_Episodes_Felony = float(data["Prior_Arrest_Episodes_Felony"])
    Age_at_Release = float(data["Age_at_Release"])
    Prior_Arrest_Episodes_Property = float(data["Prior_Arrest_Episodes_Property"])
    DrugTests_THC_Positive = float(data["DrugTests_THC_Positive"])
    Prior_Arrest_Episodes_Misd = float(data["Prior_Arrest_Episodes_Misd"])
    Prior_Conviction_Episodes_Misd = float(data["Prior_Conviction_Episodes_Misd"])
    Prior_Conviction_Episodes_Prop = float(data["Prior_Conviction_Episodes_Prop"])
    Prison_Offense_Violent_Sex = float(data["Prison_Offense_Violent_Sex"])
    Prison_Years = float(data["Prison_Years"])
    Condition_MH_SA = float(data["Condition_MH_SA"])

    vec_to_scale = np.array([[Supervision_Risk_Score_First, Prior_Arrest_Episodes_PPViolationCharges, 
                    Prior_Arrest_Episodes_Felony, Age_at_Release, Prior_Arrest_Episodes_Property, 
                    Prior_Arrest_Episodes_Misd, Prior_Conviction_Episodes_Misd, Prior_Conviction_Episodes_Prop, 
                    Prison_Years]])

    scaled_vec = list(scaler.transform(vec_to_scale)[0])

    Supervision_Risk_Score_First, Prior_Arrest_Episodes_PPViolationCharges, Prior_Arrest_Episodes_Felony, Age_at_Release, Prior_Arrest_Episodes_Property, Prior_Arrest_Episodes_Misd, Prior_Conviction_Episodes_Misd, Prior_Conviction_Episodes_Prop, Prison_Years = scaled_vec

    input_vec = np.array([[Percent_Days_Employed, Supervision_Risk_Score_First, Prior_Arrest_Episodes_PPViolationCharges, 
                Gang_Affiliated, Prior_Arrest_Episodes_Felony, Age_at_Release, Prior_Arrest_Episodes_Property, 
                DrugTests_THC_Positive, Prior_Arrest_Episodes_Misd, Prior_Conviction_Episodes_Misd, 
                Prior_Conviction_Episodes_Prop, Prison_Offense_Violent_Sex, Prison_Years, Condition_MH_SA]])
    
    # Model Prediction 
    prediction = rf.predict(input_vec)
    print(prediction[0])

    return Response(json.dumps(str(prediction[0])))
