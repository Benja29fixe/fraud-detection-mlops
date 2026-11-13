import pandas as pd

results= []

for i in range(10):

    results.append({
        "i": i,
        "precision": i*2,
        "recall": i*10,
        "f1": i**2
    })

df= pd.DataFrame(results)

print(df.head().to_string(index=False))

print(df.drop(columns=['f1']))

best_row = df.loc[
    df["f1"].idxmax()
]

print(f"{best_row =}")

####################
# MAIN
####################
if __name__ == '__main__':
    print("Hello Benjamin")