import json
from pickle import TRUE
import requests
import sys
import os
import socket
import time
import numpy as np
import h5py
import argosfeddeep.app as ap
import shutil

#__all__ = ['get_token','post_model_to_master','flush_model_folders','get_model_path']

# loggers
info = lambda msg: sys.stdout.write("info > " + msg + "\n")
warn = lambda msg: sys.stdout.write("warn > " + msg + "\n")

api_port="5050"
#proxy = "http://52.28.49.157"
#proxy='https://aggregator-vm-argos.railway.medicaldataworks.nl'
proxy='http://20.93.147.169:5050'
url_download = proxy+ "/api/download"
url_upload = proxy+ "/api/upload"

#Master part of the algorithm
def get_token():
    
    local_ip = "http://"+socket.gethostbyname(socket.gethostname())+':7000'
    print(local_ip)
    port = os.environ['API_FORWARDER_PORT']
    url = os.environ['HOST'] +":"+port+"/login"
    body = {"password": os.environ['API_FORWARDER_PASSWORD'],"local_ip":local_ip}
    headers = {'content-type': 'application/json'}

    r = requests.post(url, data=json.dumps(body), headers=headers)
    res = r.json()
    token = res['access_token']
    return token

#Node Part of the Algorithm
def post_model_to_master(params,trained_model_path,token):
    headers = { "enctype":"multipart/form-data","Authorization": "Bearer " + token}
    try:
        with open (trained_model_path,'rb') as f:
                file_dict = {"file": f}
                response = requests.post(url = url_upload, files=file_dict, params=params, headers= headers)
                time.sleep(10)
                status_code = response.status_code
                return status_code
    except:
        print("Cannot send file")
        status_code = 500
        return status_code 

#Node Part of the algorithm
def get_model_path(token,iteration):
    if not os.path.exists(os.path.join(os.getcwd(),'assets','averaged_model')):
        os.makedirs(os.path.join(os.getcwd(),'assets','averaged_model'))
    node_averaged_model_dir = os.path.join(os.getcwd(),'assets','averaged_model')
    headers = {"enctype":"multipart/form-data","Authorization": "Bearer " + token}
    while True: 
        #response.raise_for_status()
        averaged_model_name = os.path.join(node_averaged_model_dir, 'averaged_iteration_'+str(iteration)+'.h5')
        if not os.path.exists(averaged_model_name):
            response = requests.get(url_download, params = {"iteration":iteration},headers=headers, stream=True, timeout=3600)
            if response.status_code==200:
                raw_content = response.content
                hf=h5py.File(averaged_model_name,'w')
                npdata=np.array(raw_content)
                dset=hf.create_dataset(averaged_model_name,data=npdata)
                break
        else:
            print("File Not Available... Waiting")
            time.sleep(30)
            continue
    return averaged_model_name

def flush_model_folders(iteration):
    upload_folder = ap.app.config['UPLOAD_FOLDER']
    shutil.rmtree(os.path.join(upload_folder,'iteration_'+str(iteration-5)))

def flush_all_folders():
    folder1 = ap.app.config['UPLOAD_FOLDER']
    folder2 = ap.app.config['DOWNLOAD_FOLDER'] 
    shutil.rmtree(folder1)
    shutil.rmtree(folder2)

'''def check_results(client_token,ids):
    url = "https://mdw-vantage6-argos.azurewebsites.net:443"
    api_path = "/api"
    headers = {'Authorization': 'Bearer ' + client_token}
    result_status = requests.get(url + api_path+"/collaboration/"+str(ids),headers=headers)
    result_status_json = result_status.json()
    status = result_status_json['finished_at']
    if not status: return False 
    return TRUE'''
            
    



