# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# %%
df = pd.read_csv('clean_dataset.csv')

# %%
df.head()

# %%
df.isnull().sum()

# %% [markdown]
# ##### **Encode Categorical Varibles**

# %%
label_encoders={}
categorical_columns = ['Industry', 'Ethnicity', 'Citizen']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    

# %% [markdown]
# ##### **Create Features and Targets**

# %%
X = df.drop(columns=['Approved']) #The remaining are the features
y= df['Approved'] # Target variable

# %% [markdown]
# ##### **Split the data into Train and Test**

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ##### **Feature Scaling**

# %%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% [markdown]
# ### **Model Building**

# %% [markdown]
# ##### **1. Logistic Regression**

# %% [markdown]
# ###### *Train the model*

# %%
model = LogisticRegression(random_state=42)
model.fit(X_train,y_train)

# %% [markdown]
# ###### *Model Evaluation*

# %%
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# %% [markdown]
# ##### **2. Random Forest**

# %% [markdown]
# ###### *Train the Model*

# %%
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# %% [markdown]
# ###### *Model Evaluation*

# %%
y_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf))

# %% [markdown]
# ##### ROC-AUC Evaluation

# %%
y_prob_rf= rf_model.predict_proba(X_test)[:,1]
roc_auc = roc_auc_score(y_test, y_prob_rf)
print(f"ROC-AUC Score: {roc_auc:.2f}")

# %% [markdown]
# ##### **Save the Model**

# %%
joblib.dump(rf_model, 'random_forest_model.pkl')
print("Model saved as a radom_forest_model.pkl")


