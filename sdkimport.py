import previsionio as pio
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import  make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


import os
from os.path import join
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME="Sklearn models Comparison"
TRAINSET_NAME="fraud_train"
HOLDOUT_NAME="fraud_holdout"
INPUT_PATH=join("data","assets")
TARGET = 'fraude'


pio.client.init_client(
    token=os.environ['PIO_MASTER_TOKEN'],
    prevision_url=os.environ['DOMAIN'])

# Check
logging.info("List of your projects")
projects_list = pio.Project.list() 
for p in projects_list: 
    logging.info(p.name) 



# Create a new Project or using the old one

if  PROJECT_NAME not in [p.name for p in projects_list] :
    logging.info(f"Creating new project {PROJECT_NAME}")
    project = pio.Project.new(name=PROJECT_NAME,  description="An experiment using ")
else :
    logging.info(f"Reusing project {PROJECT_NAME}")
    project = [p for p in projects_list if p.name==PROJECT_NAME] [0]


logging.info(f"Uploading Datasets to {project.name}")

datasets_list = project.list_datasets()
for d in datasets_list: 
    logging.info(d.name) 


# Reuse output of sktoonx.py

if TRAINSET_NAME in [d.name for d in datasets_list] :
    logging.info("Trainset with same name already exist in this project")
    train = [d for d in datasets_list if d.name==TRAINSET_NAME] [0]
else :
    logging.info("Trainset does not exist yet. Uploading it")
    train = project.create_dataset(file_name=join(INPUT_PATH,"trainset_fraud.csv"), name='fraud_train')



if HOLDOUT_NAME in [d.name for d in datasets_list] :
    logging.info("Holdout  with same name already exist in this project")
    test = [d for d in datasets_list if d.name==HOLDOUT_NAME] [0]
else :
    logging.info("Holdout does not exist yet. Uploading it")
    test  = project.create_dataset(file_name=join(INPUT_PATH,"holdout_fraud.csv"), name='fraud_holdout')



logging.info("Dataset available.")

train_data = train.data.astype(np.float32)
test_data = test.data.astype(np.float32)

X_train = train_data.drop(TARGET, axis=1)
y_train = train_data[TARGET]



classifiers=[ {
                "name":"lrsklearn",
                "algo":LogisticRegression(max_iter=3000)
                },
                {  
                "name":"knnsk",
                "algo": KNeighborsClassifier(3)
                },
                {
                    "name":"svcskl",
                    "algo":SVC(kernel="linear", C=0.025)
                },
                {
                    "name":"svcsk",
                    "algo":SVC(gamma=2, C=1)
                },
                {
                    "name":"gpcsk",
                    "algo":GaussianProcessClassifier(1.0 * RBF(1.0))
                },
                {
                    "name":"dtsk",
                    "algo":DecisionTreeClassifier(max_depth=5),
                },
                {
                    "name":"rfsk",
                    "algo":RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                },
                {
                    "name":"mlpcsk",
                    "algo":MLPClassifier(alpha=1, max_iter=1000),
                },
                {
                    "name":"gsk",
                    "algo":GaussianNB(),
                }
            ]

logging.info("Input are all float")
initial_type = [('float_input', FloatTensorType([None,X_train.shape[1]]))]

logging.info("Saving config files")
config={}
config["class_names"] = [str(c) for c in set(y_train)]
config["input"] = [str(feature) for feature in X_train.columns]
with open(join(INPUT_PATH,'logreg_fraude.yaml'), 'w') as f:
    yaml.dump(config, f)



logging.info("Testing many Sklearn Classifiers")
for clf in classifiers :
    logging
    clr = make_pipeline(OrdinalEncoder(),clf["algo"])
    clr.fit(X_train, y_train )

    onx = convert_sklearn(clr, initial_types=initial_type)
    with open(join(INPUT_PATH,f'{clf["name"]}_logreg_fraude.onnx'), 'wb') as f:
        f.write(onx.SerializeToString())


logging.info( "Uploading all the models in the same experiment")
external_models=[(clf["name"],join(INPUT_PATH,f'{clf["name"]}_logreg_fraude.onnx'), join(INPUT_PATH,'logreg_fraude.yaml')) for clf in classifiers ]
exp = project.create_external_classification(experiment_name=f'churn_sklearn_{clf["name"]}',
                                    dataset=train,
                                    holdout_dataset=test,
                                    target_column=TARGET,
                                    external_models =  external_models
                                  )
