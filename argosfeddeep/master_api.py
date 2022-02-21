from flask import Flask, jsonify
from flask import request
import argosfeddeep.app as ta
import argosfeddeep.database as db
import os
import urllib.request
import urllib.request
from flask import Flask, request, redirect, jsonify, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from flask import send_file, send_from_directory, safe_join, abort
from pathlib import Path
import sys


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'h5'])
database = '/mnt/data/argos.db'
data_path='/mnt/data'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# loggers
info = lambda msg: sys.stdout.write("info > " + msg + "\n")
warn = lambda msg: sys.stdout.write("warn > " + msg + "\n")

''' endpoints                                 Method        Description                      
     http://public_ip:port/api/upload          POST             post from data node to the master container
     http://public_ip:port/api/download        GET              fetch model from master node
'''

app = Flask(__name__)
app.secret_key = "secret key"

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        iteration = int(request.args.get('iteration'))
        org_id = int(request.args.get('org_id'))
        nodeType = request.args.get('nodeType')
        training_loss = float(request.args.get('training_loss'))
        training_dice = float(request.args.get('training_dice'))
        validation_loss = float(request.args.get('validation_loss'))
        validation_dice = float(request.args.get('validation_dice'))
        file = request.files['file']
        if not os.path.exists(os.path.join(data_path,'upload',str(iteration))):
            os.makedirs(os.path.join(data_path,'upload',str(iteration)))
        model_path =os.path.join(data_path,'upload',str(iteration))
        filename = str(iteration)+"_"+str(org_id)+"_node.h5"
        file.save(os.path.join(model_path,filename))
        model = (nodeType,iteration,org_id,training_loss,training_dice,validation_loss,validation_dice,model_path)
        #conn = db.create_connection(database)
        #db.insert_into_table_nodeModel(conn,model)
    return {"message:":"file inserted"}
 
@app.route('/api/download', methods=['GET'])
def api_get_model():
    iteration = request.args.get('iteration')
    conn = db.create_connection(database)
    value = db.extract_from_table_aggregate(conn,int(iteration))
    return redirect(url_for('download_file', filename=value))

@app.route('/api/download/<filename>')
def download_file(filename):
    return send_from_directory('/mnt/data/download', filename, as_attachment=True)

if __name__ == "__main__":
    # app.debug = True
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=7000, debug=True)
    