import pandas as pd

data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'Air Temp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'Enjoy Sport': ['Yes', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

def find_s_algorithm(dataframe):
    features = dataframe.iloc[:, :-1].values
    target = dataframe.iloc[:, -1].values
    hypothesis = None
    for i in range(len(target)):
        if target[i] == 'Yes':
            hypothesis = features[i].copy()
            break
    for i in range(len(features)):
        if target[i] == 'Yes':
            for j in range(len(hypothesis)):
                if hypothesis[j] != features[i][j]:
                    hypothesis[j] = '?'

    return hypothesis
hypothesis = find_s_algorithm(df)
print('The most specific hypothesis is:', hypothesis)
