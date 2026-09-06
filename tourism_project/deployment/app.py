import os
import joblib
import pandas as pd
import streamlit as st

# Load the trained model committed to the repository by the pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), "tourism_model.joblib")
model = joblib.load(MODEL_PATH)

st.title("Wellness Tourism Package Prediction")
st.write(
    "This app predicts whether a customer is likely to purchase the newly introduced "
    "Wellness Tourism Package. Enter the customer details below and click Predict."
)

st.subheader("Customer Details")
age = st.number_input("Age", min_value=18, max_value=100, value=35)
typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
citytier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthlyincome = st.number_input("Monthly Income", min_value=1000.0, max_value=100000.0, value=20000.0)
passport = st.selectbox("Has Passport?", ["Yes", "No"])
owncar = st.selectbox("Owns a Car?", ["Yes", "No"])

st.subheader("Trip Details")
numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=3)
numberofchildrenvisiting = st.number_input("Number of Children Visiting (below age 5)", min_value=0, max_value=5, value=1)
preferredpropertystar = st.selectbox("Preferred Property Star Rating", [3.0, 4.0, 5.0])
numberoftrips = st.number_input("Average Number of Trips per Year", min_value=0, max_value=25, value=3)

st.subheader("Sales Interaction Details")
productpitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=1.0, max_value=60.0, value=15.0)
numberoffollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=4)
pitchsatisfactionscore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

# Collect all the inputs into a single row dataframe in the same format as the training data
input_data = pd.DataFrame([{
    "Age": float(age),
    "TypeofContact": typeofcontact,
    "CityTier": int(citytier),
    "DurationOfPitch": float(durationofpitch),
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": int(numberofpersonvisiting),
    "NumberOfFollowups": float(numberoffollowups),
    "ProductPitched": productpitched,
    "PreferredPropertyStar": float(preferredpropertystar),
    "MaritalStatus": maritalstatus,
    "NumberOfTrips": float(numberoftrips),
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": int(pitchsatisfactionscore),
    "OwnCar": 1 if owncar == "Yes" else 0,
    "NumberOfChildrenVisiting": float(numberofchildrenvisiting),
    "Designation": designation,
    "MonthlyIncome": float(monthlyincome)
}])

st.subheader("Entered Customer Data")
st.write(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"This customer is likely to PURCHASE the Wellness Tourism Package. (Probability: {probability:.2f})")
    else:
        st.warning(f"This customer is NOT likely to purchase the Wellness Tourism Package. (Probability: {probability:.2f})")
