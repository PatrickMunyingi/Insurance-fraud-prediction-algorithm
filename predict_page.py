import streamlit as st
import numpy as np
import pickle

def load_model():
    with open('saved_steps.pkl','rb')as file:
        data=pickle.load(file)
    return data
data=load_model()
st.title("**We will need some infomation**")
st.sidebar.title("**Fraudulent claim prediction web app**")

lr=data['model']
le_policy_state=data['le_policy_state']
le_policy_csl=data['le_policy_csl']
le_sex=data['le_sex']
le_education=data['le_education']
le_occupation=data['le_occupation']
le_hobbies=data['le_hobbies']
le_relationship=data['le_relationship']
le_incident_type=data['le_incident_type']
le_collision_type=data['le_collision_type']
le_incident_severity=data['le_incident_severity']
le_incident_state=data['le_incident_state']
le_property_damage=data['le_property_damage']
le_auto_make=data['le_auto_make']

def show_predict_page():
    st.sidebar.title("Fradulent claims detection web app")


    st.write("""**### We will need some info:**""")
policy_state=('Ohio','Illinois','Indiana')
policy_csl=('250/500', '100/300', '500/1000')
sex=('MALE', 'FEMALE')
education=('MD', 'PhD', 'Associate', 'Masters', 'High School', 'College','JD')
occupation=('craft-repair', 'machine-op-inspct', 'sales', 'armed-forces',
       'tech-support', 'prof-specialty', 'other-service',
       'priv-house-serv', 'exec-managerial', 'protective-serv',
       'transport-moving', 'handlers-cleaners', 'adm-clerical',
       'farming-fishing')
hobbies=('sleeping', 'reading', 'board-games', 'bungie-jumping',
       'base-jumping', 'golf', 'camping', 'dancing', 'skydiving',
       'movies', 'hiking', 'yachting', 'paintball', 'chess', 'kayaking',
       'polo', 'basketball', 'video-games', 'cross-fit', 'exercise')
relationship=('husband', 'other-relative', 'own-child', 'unmarried', 'wife',
       'not-in-family')
incident_type=('Single Vehicle Collision', 'Vehicle Theft',
       'Multi-vehicle Collision', 'Parked Car')
collision_type=('Side Collision', '?', 'Rear Collision', 'Front Collision')
incident_severity=('blank', 'Minor Damage', 'Major Damage', 'Total Loss',
       'Trivial Damage')
incident_state=('South Carolina', None, 'NewYork', 'Ohio', 'West Virginia',
       'North Carolina', 'Pennsylvania')
property_damage=('YES', '?', 'NO')
auto_make=('Saab', 'Mercedes', 'Dodge', 'Chevrolet', 'Accura', 'Nissan',
       'Audi', 'Toyota', 'Ford', 'Suburu', 'BMW', 'Jeep', 'Honda',
       'Volkswagen')

policy_states=st.selectbox('Policy State',policy_state)
policy_csls=st.selectbox('Policy csl',policy_csl)
gender=st.sidebar.selectbox('Gender',sex)
education_level=st.selectbox('Education Level',education)
occupation_level=st.selectbox('Occupation',occupation)
hobby=st.selectbox('Hobby',hobbies)
relationship_level=st.selectbox('Relationship Level',relationship)
incident_type_=st.selectbox('Incident Type',incident_type)
collision=st.selectbox('Collision Type',collision_type)
severity=st.selectbox('Incident Severity',incident_severity)
incident_state_=st.selectbox('Incident State',incident_state)
property=st.selectbox('Property Damage',property_damage)
auto=st.selectbox('Auto Make',auto_make)

years=st.sidebar.number_input('Years as customers',0,50,0)
age=st.sidebar.slider('Age',0,100,0)
policy_deductable=st.sidebar.number_input('Policy Deductable',0,10000000)
annual_premium=st.sidebar.number_input('Annual Premium',0,1000000000)
vehicle_number=st.sidebar.number_input('Number of vehicles inolved in accident',0,1000)
injuries=st.sidebar.slider('Persons injured',0,100)
injury_claim=st.sidebar.number_input('Injury Claims',0,10000000)
property_claim=st.sidebar.number_input('Property Claim',0,1000000000000)
vehicle_claim=st.sidebar.number_input('Vehicle claim',0,10000000000)
auto_year=st.slider('Year car was made',1980,2032)


ok=st.button('Proceed with classification')
from sklearn.preprocessing import LabelEncoder

# ...

if ok:
    x = np.array([[policy_states, policy_csls, gender, education_level, occupation_level, hobby, relationship_level, incident_type_, collision,
                   severity, incident_state_, property, auto, years, age, policy_deductable, annual_premium, vehicle_number, injuries,
                   injury_claim, property_claim, vehicle_claim, auto_year]])

    le = LabelEncoder()

    x = le.fit_transform(x.ravel())

    x = x.astype(float)
    x = x.reshape(1, -1)

    prediction = lr.predict(x)
    if prediction[0] == 0:
        st.subheader("The claim might not be fraudulent")
    else:
        st.subheader("The claim might be a result of fraud")
