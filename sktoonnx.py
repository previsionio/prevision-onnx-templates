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
import yaml



logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

logging.info("This script builds a classifier with sklearn and save it to onnx format ")
logging.info("it generates the onnx file, the config file, trainset and holdout for reproducibility")

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


logging.info("Rebuilding the final dataset")
ff = Xcat.join(pd.DataFrame(Xnum,columns=num_feat, index=Xcat.index))
ff.replace([np.inf, -np.inf], np.nan, inplace=True)
ff = ff.fillna(-1)

logging.info("Converting all data to Numpy Float 32")
logging.info("in order to have a Onnx modele with only one data type")
ff = ff.astype(np.float32)
logging.info("The dataset is an array of Float 32")


logging.info( "Split the dataset and save for holdout validation")
X_train, X_test, y_train, y_test = train_test_split(ff, y, test_size=0.2, random_state=754)
X_train.join(y_train).to_csv(join(OUTPUT_PATH,'trainset_fraud.csv'))
X_test.join(y_test).to_csv(join(OUTPUT_PATH,'holdout_fraud.csv'))


logging.info("********************** Modelisation ***********************************")
clf =  RandomForestClassifier(max_depth=50,verbose=1, n_estimators=200, max_features=1)
clf.fit(X_train, y_train)
clf.score(X_train, y_train)

logging.info("********************** Validation ***********************************")
# Check its not that bad
yhat=clf.predict(X_test)


cm = confusion_matrix(y_test, yhat)
logging.info("Original Model Confusion matrix")
logging.info(cm[0])
logging.info(cm[1])

logging.info("********************** Converting to onnx ***********************************")
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

logging.info("Datas are a tensor of float_input whose shape is [None, X_train numbers of columns without target]")
initial_type = [('float_input', FloatTensorType([None,  np.array(X_train).shape[1]]))]
logging.info("Converting")
onx = convert_sklearn(clf, initial_types=initial_type)

logging.info(f"Saving the model to {join(OUTPUT_PATH,'classif_fraud.onnx')}")
with open(join(OUTPUT_PATH,"classif_fraud.onnx"), "wb") as f:
    f.write(onx.SerializeToString())

logging.info("Generating and saving config file")
config={}
config["class_names"] = [str(c) for c in set(y_train)]
config["input"] = [str(feature) for feature in X_train.columns]

with open(join(OUTPUT_PATH,"classif_fraud_config.yaml"), 'w') as file:
    documents = yaml.dump(config, file)




logging.info("********************** Checking the onnx file ***********************************")
import onnxruntime as rt

logging.info("Loading a previous model")
sess = rt.InferenceSession(join(OUTPUT_PATH,"classif_fraud.onnx"))

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

logging.info("Applyed it on testest")
cols = list(pd.read_csv(join(OUTPUT_PATH,"holdout_fraud.csv"), nrows =1))
X_test=np.array(pd.read_csv(join(OUTPUT_PATH,'holdout_fraud.csv'), usecols =[i for i in cols if i != 'fraude']).set_index('TransactionID'))


pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
cm = confusion_matrix(y_test, pred_onx)
logging.info("Onnx Model Confusion matrix")
logging.info(cm[0])
logging.info(cm[1])

logging.info("Check the values. Everything ran fine.")
logging.info("You can grab the model, its configuration file, a trainset and holdout in ")
logging.info(OUTPUT_PATH)
exit(0)