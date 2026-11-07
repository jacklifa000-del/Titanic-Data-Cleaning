import pandas as pd  

#load dataset
df = pd.read_csv("Titanic Dataset.csv")

#print the first five rows
print(df.head())

print("\n--- INFO---")
print(df.info())

print("\n--- MISSING VALUES----")
print(df.isnull().sum())

# fill in missing ages with mean
df["age"] = df["age"].fillna(df["age"].mean())

# drop columns that are too messy(cabin, body, boat, home.dest)
df = df.drop(columns=["cabin", "boat", "body", "home.dest"])

print("\n--- After clean----")
print(df.isnull().sum())

# fill in missing FARE with mean
df["fare"] = df["fare"].fillna(df["fare"].mean())

#  fill in missing Embars with mean
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])


print("\n--- final check----")
print(df.isnull().sum())

# drop columns that are too messy(name,ticket)
df = df.drop(columns=["name","ticket"])


#encode
df["sex"]=df["sex"].map({"male": 0, "female": 1})

#one hot encode
df = pd.get_dummies(df, columns=["embarked"], drop_first=True)

print("\n---FINAL ENCODED DATA")
print(df.head())


from sklearn.preprocessing import StandardScaler

#create thr scaler
scaler=StandardScaler()

#scale the age and fare columns
df[["age","fare"]] = scaler.fit_transform(df[["age","fare"]])

print("SCALED")
print(df.head)

import torch
tensor_data=torch.tensor(df.values.astype(float), dtype=torch.float32)
print("tensor shape")
print(tensor_data.shape)

df=df.reset_index(drop=True)

df.to_csv("Titanic_Cleaned.csv", index=False)

print("File saved successfully!")

