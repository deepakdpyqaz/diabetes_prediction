import subprocess
subprocess.call('sudo docker stop $(sudo docker ps -a -q)', shell=True)
subprocess.call('sudo docker rm $(sudo docker ps -a -q)', shell=True)