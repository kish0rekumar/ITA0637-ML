import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Step 1: Reading the dataset
data = pd.DataFrame({
    'battery_power': [842, 1021, 563, 615, 1821, 1859, 1821, 1954, 1445, 509],
    'blue': [0, 1, 1, 0, 0, 1, 1, 0, 1, 1],
    'clock_speed': [2.2, 0.5, 0.5, 2.5, 1.2, 0.5, 1.0, 0.5, 1.7, 0.5],
    'dual_sim': [0, 1, 1, 0, 1, 0, 1, 1, 1, 1],
    'fc': [1, 0, 2, 13, 3, 4, 0, 3, 0, 0],
    'four_g': [0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    'int_memory': [7, 53, 41, 10, 44, 22, 4, 53, 22, 46],
    'm_dep': [0.6, 0.7, 0.9, 0.2, 0.5, 0.3, 0.4, 0.1, 0.8, 0.1],
    'mobile_wt': [188, 136, 145, 131, 141, 164, 139, 187, 174, 93],
    'n_cores': [2, 3, 5, 6, 2, 3, 5, 6, 2, 5],
    'pc': [2, 6, 6, 9, 14, 7, 0, 16, 10, 6],
    'px_height': [20, 905, 1263, 1216, 1208, 1004, 381, 512, 1988, 754],
    'px_width': [756, 1988, 1716, 1786, 1215, 1654, 1366, 1028, 858, 1784],
    'ram': [2549, 2631, 2603, 2769, 1411, 1067, 3220, 700, 1099, 513],
    'sc_h': [9, 17, 11, 16, 8, 13, 17, 19, 11, 16],
    'sc_w': [7, 3, 2, 8, 2, 8, 1, 10, 0, 7],
    'talk_time': [19, 7, 9, 11, 15, 10, 18, 5, 19, 2],
    'three_g': [0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    'touch_screen': [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    'wifi': [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    'price_range': [1, 2, 2, 2, 1, 1, 1, 3, 0, 0]
})

# Step 2: Printing the first five rows
print(data.head())

# Step 3: Basic statistical computations
print(data.describe())
# Step 4: Identifying columns and their data types
print(data.info())

# Step 5: Detecting and handling null values
print(data.isnull().sum())

# Replacing null values with mode
for column in data.columns:
    if data[column].isnull().sum() > 0:
        mode_value = data[column].mode()[0]
        data[column].fillna(mode_value, inplace=True)

print(data.isnull().sum())

# Step 6: Exploring the dataset using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Step 7: Splitting the data into training and testing sets
X = data.drop('price_range', axis=1)
y = data['price_range']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Fitting the Naive Bayes Classifier model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 9: Predicting with the model
y_pred = model.predict(X_test)

# Step 10: Finding the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the Naive Bayes Classifier: {accuracy:.2f}')
