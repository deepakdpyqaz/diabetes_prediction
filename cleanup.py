import subprocess
import os
from absl import logging
logging.set_verbosity(logging.INFO)  

logging.info(os.getcwd())
if not os.path.exists("serving_model"):
    os.mkdir("serving_model")
if not os.path.exists("serving_model/dibetes_prediction_pipeline"):
    os.mkdir("serving_model/dibetes_prediction_pipeline")
logging.info(os.listdir())
logging.info(os.listdir("serving_model"))
logging.info(os.listdir("serving_model/dibetes_prediction_pipeline"))
subprocess.call('sudo pwd',shell=True)
subprocess.call('sudo ls',shell=True)
subprocess.call('sudo ls serving_model/',shell=True)
subprocess.call('sudo ls serving_model/dibetes_prediction_pipeline',shell=True)
# subprocess.call('sudo cp -r /home/deepak/pipeline/serving_model/dibetes_prediction_pipeline/* /home/deepak/diabetes_prediction/serving_model/dibetes_prediction_pipeline/',shell=True)
# subprocess.call('sudo docker stop $(sudo docker ps -a -q)', shell=True)
# subprocess.call('sudo docker rm $(sudo docker ps -a -q)', shell=True)
