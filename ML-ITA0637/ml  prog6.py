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


examples = df.values.tolist()

# Step 2: Initialize the general (G) and specific (S) hypotheses
def initialize_hypotheses(examples):
    specific_h = examples[0][:-1].copy()
    general_h = [['?' for _ in range(len(specific_h))] for _ in range(len(specific_h))]
    return specific_h, general_h

def consistent(hypothesis, example):
    for h, e in zip(hypothesis, example):
        if h != '?' and h != e:
            return False
    return True

def candidate_elimination(examples):
    specific_h, general_h = initialize_hypotheses(examples)
    print(f"Initial Specific Hypothesis: {specific_h}")
    print(f"Initial General Hypothesis: {general_h}")

    for i, example in enumerate(examples):
        if example[-1] == 'Yes':
            for j in range(len(specific_h)):
                if not consistent(specific_h, example[:-1]):
                    specific_h[j] = '?' if specific_h[j] != example[j] else specific_h[j]
            for g in general_h:
                if not consistent(g, example[:-1]):
                    general_h.remove(g)
        else:
            new_general_h = []
            for g in general_h:
                for j in range(len(specific_h)):
                    if g[j] == '?':
                        for value in df.iloc[:, j].unique():
                            if value != specific_h[j]:
                                new_g = g.copy()
                                new_g[j] = value
                                if consistent(new_g, specific_h):
                                    new_general_h.append(new_g)
            general_h = new_general_h.copy()

        print(f"\nExample {i + 1} processed")
        print(f"Specific Hypothesis: {specific_h}")
        print(f"General Hypothesis: {general_h}")

    return specific_h, general_h

# Step 3: Apply the Candidate-Elimination algorithm to the dataset
specific_h, general_h = candidate_elimination(examples)

# Step 4: Output the final version space
print("\nFinal Version Space:")
print(f"Specific Hypothesis: {specific_h}")
print(f"General Hypothesis: {general_h}")

     
