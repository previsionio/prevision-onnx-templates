import pandas as pd 
from os.path import join
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
import numpy as np
import logging

from pathlib import Path



logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


logging.info("Loading a small dataset for test")
PATH="data"
INPUT_PATH=join(PATH, "transaction_fraud_subset.csv")
OUTPUT_PATH=join(PATH,"assets")

Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH).set_index('TransactionID')


logging.info("********************** Feature Engineering ***********************************")
logging.info("Dropping unused featured for this simple model")
X=df.drop(columns=["fraude","DeviceInfo","Distance Facturation"])
logging.info("Putting target aside")
y=df["fraude"]


logging.info("Splitting Numerical features and categorical one")
cat_feat=["DeviceType",
            "Code Produit",
            "Domaine Email Login",
            "Domaine email livraison",
            ]
num_feat = [feat for feat in X.columns if feat not in cat_feat]


logging.info("Simple dummy encoding for embedding Categorical features")
Xcat = pd.get_dummies(X[cat_feat])

logging.info("Simple scaler for numerical features")
scaler = StandardScaler()
Xnum = scaler.fit_transform(X[num_feat])


logging.info
ff = Xcat.join(pd.DataFrame(Xnum,columns=num_feat, index=Xcat.index))
ff.replace([np.inf, -np.inf], np.nan, inplace=True)
ff = ff.fillna(-1)

#Lets convert everything to float for easier onnx management
ff = ff.astype(np.float32)



X_train, X_test, y_train, y_test = train_test_split(ff, y, test_size=0.2, random_state=754)

# Save for later use
X_train.join(y_train).to_csv(join(OUTPUT_PATH,'trainset_fraud.csv'))
X_test.join(y_test).to_csv(join(OUTPUT_PATH,'testset_fraud.csv'))


logging.info("********************** Modelisation ***********************************")
clf =  RandomForestClassifier(max_depth=50,verbose=1, n_estimators=200, max_features=1)
clf.fit(X_train, y_train)
clf.score(X_train, y_train)

logging.info("********************** Validation ***********************************")
# Check its not that bad
yhat=clf.predict(X_test)
confusion_matrix(y_test, yhat)



logging.info("********************** Converting to onnx ***********************************")
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


initial_type = [('float_input', FloatTensorType([None,  np.array(X_train).shape[1]]))]
onx = convert_sklearn(clf, initial_types=initial_type)

with open(join(OUTPUT_PATH,"classif_fraud.onnx"), "wb") as f:
    f.write(onx.SerializeToString())




logging.info("********************** Checking the onnx file ***********************************")
import onnxruntime as rt
import numpy
sess = rt.InferenceSession(join(OUTPUT_PATH,"classif_fraud.onnx"))

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

cols = list(pd.read_csv(join(OUTPUT_PATH,"testset_fraud.csv"), nrows =1))
X_test=np.array(pd.read_csv(join(OUTPUT_PATH,'testset_fraud.csv'), usecols =[i for i in cols if i != 'fraude']).set_index('TransactionID'))


pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]