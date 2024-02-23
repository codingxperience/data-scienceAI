import pandas as pd
import seaborn as sns
import numpy as np

from  sklearn.ensemble import IsolationForest
data = pd.read_excel('Py4stat.xlsx')

clean_data = data[['Bio','Che']] = data[['Bio','Che']].apply(lambda x:x.str.split(' ').str[1])
#print(clean_data.value_counts(subset = ['Bio', 'Che']))

clean_data.fillna(0)

converted_data = clean_data.astype('int')
#print(converted_data.info())

sns.boxplot(clean_data.Bio)

random_state = np.random.RandomState(42)
model = IsolationForest(n_estimators=100,max_samples='auto',contamination=float(0.2),random_state=random_state)
model.fit(clean_data[['Bio', 'Che']])
print(model.get_params())

clean_data['score'] = model.decision_function(clean_data[['Bio', 'Che']])
clean_data['anormally_score'] = model.predict(clean_data[['Bio', 'Che']])
print(clean_data[clean_data['anormally_score'] == -1].head())

#Negative results mean that there are no anormalies.

#Assignment
#1. Drop columns E,F,G using python
#2.Do anormalies detection. Find the abnormal columns and the ones with the most abnormalities
#3. Find the year with the least bags.
#Find the state with the least bags.
#Find the state with the higgest bagsof avocado
#Find the state with the highest price of avocado
#Find the state that consumes the most avocado
#How may default data types are in the file and state them.

#Model evaluation - 16th January 2024