

import pandas as pd
address = 'HAM10000_metadata.csv'
df = pd.read_csv(address)
df
df.info()
df.isna().sum()
df['dx'].value_counts()
df['dx_type'].value_counts()
dx_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
df['diagnosis'] = df['dx'].map(dx_dict.get) 
df['diagnosis'].value_counts()
df['sex'].value_counts()
df['localization'].value_counts()
df['age'].value_counts()
from sklearn.impute import SimpleImputer
import numpy as np
imputer = SimpleImputer(missing_values= np.nan,strategy='mean')  
Car_impute = imputer.fit(df[['age']])
df['age'] = Car_impute.transform(df[['age']]).ravel()
df.isna().sum()
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df1 = df.copy()
lesion_id_cat = label_encoder.fit_transform(df1['lesion_id'])
lesion_id_cat = pd.DataFrame({'lesion_id_cat': lesion_id_cat})
image_id_cat = label_encoder.fit_transform(df1['image_id'])
image_id_cat = pd.DataFrame({'image_id_cat': image_id_cat})
dx_cat = label_encoder.fit_transform(df1['dx'])
dx_cat = pd.DataFrame({'dx_cat': dx_cat})
dx_type_cat = label_encoder.fit_transform(df1['dx_type'])
dx_type_cat = pd.DataFrame({'dx_type_cat': dx_type_cat})
sex_cat = label_encoder.fit_transform(df1['sex'])
sex_cat = pd.DataFrame({'sex_cat': sex_cat})
localization_cat = label_encoder.fit_transform(df1['localization'])
localization_cat = pd.DataFrame({'localization_cat': localization_cat})
diagnosis_cat = label_encoder.fit_transform(df1['diagnosis'])
diagnosis_cat = pd.DataFrame({'diagnosis_cat': diagnosis_cat})
df1.lesion_id = lesion_id_cat
df1.image_id = image_id_cat
df1.dx = dx_cat
df1.dx_type = dx_type_cat
df1.sex = sex_cat
df1.localization = localization_cat
df1.diagnosis = diagnosis_cat
df1
from sklearn.preprocessing import StandardScaler
scaled_features = df1.copy()
col_names = ['lesion_id', 'image_id' , 'dx', 'dx_type', 'age', 'sex', 'localization', 'diagnosis']
features = scaled_features[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features[col_names] = features
scaled_features
X = scaled_features.drop(columns=['diagnosis'],axis=1)
from sklearn.model_selection import train_test_split
y = df.dx
y = [1 if each == 'bkl' or each == 'nv' or each == 'df' else 0 for each in df.dx]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
from sklearn.svm import SVC
model_svc = SVC()
model_svc.fit(X_train, y_train)
model_svc.score(X_train,y_train)
model_svc.score(X_test,y_test)
y_predict = model_svc.predict(X_test)
from sklearn.metrics import classification_report , confusion_matrix
import numpy as np
cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
confusion
print(classification_report(y_test, y_predict))
                        