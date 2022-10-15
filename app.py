import streamlit as st
st.set_page_config(layout="wide")

st.title('Passenger satisfaction on Shinkansen Bullet Train')

st.header('About')
st.write('This is a report on passanger satisfaction that will show insights into the customers, point to significant areas of improvement for most gain, and model Overall Expereince of Passanger on the Shinkansen Bullet Train')
st.subheader('Variable Dictionary:')
st.write('survey data')
st.write({'ID':'Identification of the passenger taking the survey',
    'Overall_Experience':'the Target Variable reported as \'satisfied\' or \'unsatisfied\' for the service',
    'Seat_Comfort':'Tiered Rating',
    'Seat_Class':'Survey takers Seat Class \'Ordinary\' or \'Green Car\'',
    'Arrival_time_convenient':'Tiered Rating of Arrival Convenience',
    'Catering':'Tiered Rating of onboard Catering',
    'Platform_Location':'Tiered Rating of Platform Location Convenience',
    'Onboard_Wifi_Service':'Tiered Rating',
    'Onboard_Entertainment':'Tierd Rating',
    'Online_Support':'Tiered Rating of Online Support encompassing booking assistance and help',
    'Ease_of_Online_Booking':'Tiered Rating',
    'Onboard_service':'Tiered Rating of Staff service onboard the train',
    'Legroom':'Tiered Rating',
    'Baggage_Handling':'Tiered Rating',
    'CheckIn_Service':'Tiered Rating',
    'Cleanliness':'Tiered Rating of the onboard services cleanliness',
    'Online_Boarding':'Tiered Rating of the Online Boarding Experience'})

st.write('passenger data')
st.write({'ID':'Identification of the Passanger',
    'Gender':'reported Gender',
    'Customer_Type':'Passenger Loyalty levels',
    'Age':'Age of Passanger',
    'Type_Travel':'Reason for Travel',
    'Travel_Class':'Passenger Travel Class',
    'Travel_Distance':'Distance being Traveled',
    'Departure_Delay_in Mins':'Delay of Departure',
    'Arrival_Delay_in_Mins':'Delay of Arrival'
    })

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.header('EDA')

TrainingSurvey = pd.read_csv('Surveydata_train.csv')
with st.expander("See Survey DataFrame"):
    st.dataframe(TrainingSurvey)
TrainingTravel = pd.read_csv('Traveldata_train.csv')
with st.expander("See Passenger DataFrame"):
    st.dataframe(TrainingTravel)

st.write('no duplicates were found but missing values found across survey and passenger data')
    
st.subheader('Missing Values')
col1, col2 = st.columns(2)
with col1:
    st.write('Survey Data')#Write in title and cations
    fig, ax = plt.subplots()
    ax.bar(TrainingSurvey.columns,[TrainingSurvey[col].isnull().sum()/TrainingSurvey.shape[0] for col in TrainingSurvey.columns])
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    st.pyplot(fig)

with col2:
    st.write('Passanger Data')#Write in title and cations
    fig, ax = plt.subplots()
    ax.bar(TrainingTravel.columns,[TrainingTravel[col].isnull().sum()/TrainingTravel.shape[0] for col in TrainingTravel.columns])
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    st.pyplot(fig)

st.write('for some variable there appears to be a significant number of missing values to be investigated further with Variable analysis')

st.subheader('Variable Analysis')

raw_df=TrainingTravel.join(TrainingSurvey.set_index('ID'), on='ID')

univariable_option = st.selectbox('plot variable',
    raw_df.columns)

if univariable_option in ['ID','Age','Departure_Delay_in_Mins','Travel_Distance','Arrival_Delay_in_Mins']:
    st.pyplot(sns.displot(x=raw_df[univariable_option],kind="kde"))
else:
    fig, ax = plt.subplots()
    sns.countplot(x=raw_df[univariable_option].fillna('Missing'))
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    st.pyplot(fig)

import numpy as np
transformed = np.log(raw_df[['Departure_Delay_in_Mins','Arrival_Delay_in_Mins']]+1)
st.pyplot(sns.displot(data = transformed,kind = 'kde'))
        
binned = pd.DataFrame(np.digitize(transformed, bins=[1,3]))#Binned into No Delay, Delay, Very Delayed
st.pyplot(sns.displot(data = binned,kind = 'kde'))


#Get nan Dummies
drop_col = list(filter(lambda i: i not in ['ID','Overall_Experience','Age','Travel_Distance','Departure_Delay_in_Mins','Arrival_Delay_in_Mins'], raw_df.columns))
one_hot_encoded_data = pd.get_dummies(raw_df,columns = drop_col,dummy_na = True).drop(columns=['Travel_Class_nan','Seat_Class_nan'])
one_hot_encoded_data[['Age_nan','Departure_Delay_in_Mins_nan','Arrival_Delay_in_Mins_nan']]=one_hot_encoded_data[['Age','Departure_Delay_in_Mins','Arrival_Delay_in_Mins']].isnull().astype(int)

nan_col = list(filter(lambda i: i[-3:] == 'nan', one_hot_encoded_data))

#make cat data Ordinal
standard_6tier_ratings = {'Extremely Poor':0,'Poor':1,'Needs Improvement':2,  'Acceptable':3, 'Good':4, 'Excellent':5}
ordinal_df=raw_df.replace({'Gender': {'Male': 0, 'Female': 1},
                'Customer_Type': {'Loyal Customer': 1, 'Disloyal Customer': 0},
                'Type_Travel' : {'Personal Travel':0, 'Business Travel':1},
                'Travel_Class':  {'Business':0, 'Eco':1},
                'Seat_Comfort': standard_6tier_ratings,  
                'Seat_Class' :{'Green Car':0, 'Ordinary':1},
                'Arrival_Time_Convenient': standard_6tier_ratings,
                'Catering' :standard_6tier_ratings,
                'Platform_Location' :{'Very Inconvenient':0,'Inconvenient':1,'Needs Improvement':2,'Manageable':3,'Convenient':4,'Very Convenient':5},
                'Onboard_Wifi_Service': standard_6tier_ratings,
                'Onboard_Entertainment': standard_6tier_ratings,
                'Online_Support' :standard_6tier_ratings,
                'Ease_of_Online_Booking':standard_6tier_ratings, 
                'Onboard_Service':standard_6tier_ratings,
                'Legroom': standard_6tier_ratings,
                'Baggage_Handling': {'Needs Improvement':1, 'Poor':0, 'Excellent':4, 'Acceptable':2, 'Good':3 },
                'CheckIn_Service' :standard_6tier_ratings,
                'Cleanliness' : standard_6tier_ratings,
                'Online_Boarding': standard_6tier_ratings
                }
    )
#replace skew with transform skew data    
ordinal_df[['Departure_Delay_in_Mins','Arrival_Delay_in_Mins']] = binned

#mean imputation
column_means = ordinal_df.mean()
ordinal_df = ordinal_df.fillna(column_means)
data = ordinal_df.copy()

#add back in nan columns
for col in nan_col:
    data[col]=one_hot_encoded_data[col]

fig, ax = plt.subplots(figsize=(30,20))
sns.heatmap(data.corr(), annot=True, fmt='0.2f')
st.pyplot(fig)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
transformed_data = scaler.fit_transform(ordinal_df )
transformed_df = pd.DataFrame(transformed_data, columns = ordinal_df .columns)
transformed_df['Overall_Experience']=ordinal_df ['Overall_Experience']

fig, ax = plt.subplots(figsize=(30,20))
sns.boxplot(data = transformed_df)
for tick in ax.get_xticklabels():
        tick.set_rotation(90)
st.pyplot(fig)

st.write('The Barplots show us the individual distribution of the data, the heat map shows data correlations and the variables boxplot graphic show the scaled variable distributions. It is noted that of the individual distributions many had significant number of missing values, which was investigated. The results of that investigation was that mean value imputation of the missing values would be sufficent as the missing values where randomly missing, apart from other missing values in variables. The imbalance in the data was not taken into account. It is also noted that the skew in Arrival_Delay_in_Mins and Departure_Delay_in_mins was significant enough to look at transforming it with log, revealing a bimodal distribution. Of the heatmap it should be noted that many variables showed middling to strong covariance with each other. Noting on the boxplot graphic, outlier can be seen.')





st.header('Model Building')

import scipy.stats as stats
#find absolute value of z-score for each observation
z = np.abs(stats.zscore(transformed_df))

#only keep rows in dataframe with all z-scores less than absolute value of 3 
data_clean = transformed_df[(z<3).all(axis=1)]

from sklearn.model_selection import train_test_split

y = data_clean['Overall_Experience']
X = data_clean.drop('Overall_Experience',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.tree import DecisionTreeClassifier
best_dt = DecisionTreeClassifier(criterion= 'gini', max_depth = 5,min_samples_split=50).fit(X_train,y_train)

# from joblib import dump, load
# best_dt = load('best_dt.joblib') 
y_pred = best_dt.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm/sum(cm), annot=True, fmt='0.2f')
st.pyplot(fig)

import sklearn.utils

dataset = sklearn.utils.Bunch(data=X_train, target=y_train)

from dtreeviz.trees import *
viz = dtreeviz(best_dt, 
               dataset.data,
               dataset.target,
               target_name='Overall_Experience',
               feature_names=X_train.columns,
               orientation ='LR',
               class_names=["Unsatisfied","Satified"])


import streamlit.components.v1 as components
def st_dtree(plot, height=None, width=None):

    dtree_html = f"<body>{viz.svg()}</body>"

    components.html(dtree_html, height=height, width = width)
st_dtree(viz,4200,4000)