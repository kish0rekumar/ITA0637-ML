import pandas as pd
data = {
    'Origin': ['Japan', 'Japan', 'Japan', 'USA', 'Japan'],
    'Manufacturer': ['Honda', 'Toyota', 'Toyota', 'Chrysler', 'Honda'],
    'Color': ['Blue', 'Green', 'Blue', 'Red', 'White'],
    'Decade': ['1980', '1970', '1990', '1980', '1980'],
    'Type': ['Economy', 'Sports', 'Economy', 'Economy', 'Economy'],
    'Example Type': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive']
}
df = pd.DataFrame(data)
def find_s_algorithm(df):
    hypothesis = ['ϕ'] * (df.shape[1] - 1)
    for i in range(df.shape[0]):
        if df.iloc[i]['Example Type'] == 'Positive':
            if hypothesis == ['ϕ'] * (df.shape[1] - 1):
                hypothesis = list(df.iloc[i][:-1])
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j] != df.iloc[i, j]:
                        hypothesis[j] = '?'
    return hypothesis
hypothesis = find_s_algorithm(df)
print("The most specific hypothesis is:", hypothesis)
