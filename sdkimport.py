import previsionio as pio
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


import os
from os.path import join
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME="Onnx Pipeline Import"
TRAINSET_NAME="fraud_train"
HOLDOUT_NAME="fraud_holdout"
INPUT_PATH=join("data","assets")

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
        
logging.info("Done.")

train_data = train.data
test_data = test.data
print(train_data)
print(test_data)
exit()
TARGET = 'fraude'
print(train.data)
clr = make_pipeline(OrdinalEncoder(), LogisticRegression())
clr.fit(train_data.drop(TARGET, axis=1), train_data[TARGET])

exit()

initial_type = [('float_input', FloatTensorType([None, train_data.drop(TARGET, axis=1).shape[1]]))]
onx = convert_sklearn(clr, initial_types=initial_type)
with open(join(INPUT_PATH,'logreg_fraude.onnx'), 'wb') as f:
    f.write(onx.SerializeToString())

yaml_dict = {
    'class_names': train_data[TARGET].unique().tolist(),
    'input': list(train_data.drop(TARGET, axis=1).keys())
}

with open('logreg_churn.yaml', 'w') as f:
    yaml.dump(yaml_dict, f)

p = pio.Project(_id='60d36081b4efdf001c68feb5', name='playground')

exp = p.create_external_classification(experiment_name='churn',
                                       dataset=train,
                                       holdout_dataset=test,
                                       target_column=TARGET,
                                       external_models=[
                                           ('logreg', 'logreg_churn.onnx', 'logreg_churn.yaml')
                                       ])
