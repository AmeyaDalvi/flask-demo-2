import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle


#Load csv file

df = pd.read_csv("IRIS.csv")

#Select dependant and independant variables

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

#Train-test split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

#Scaling features

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# model instantiation

classifier = RandomForestClassifier()

# fit model

classifier.fit(X_train, y_train)


#Make a pickle file for the trained model

pickle.dump(classifier, open("model.pkl", 'wb'))