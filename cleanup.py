import subprocess
subprocess.call('sudo cp -r /home/deepak/pipeline/serving_model/dibetes_prediction_pipeline/* /home/deepak/diabetes_prediction/serving_model/dibetes_prediction_pipeline/',shell=True)
subprocess.call('sudo docker stop $(sudo docker ps -a -q)', shell=True)
subprocess.call('sudo docker rm $(sudo docker ps -a -q)', shell=True)
