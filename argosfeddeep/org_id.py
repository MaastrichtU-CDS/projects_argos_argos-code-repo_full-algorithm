import os, sys
import jwt
from vantage6.client import ContainerClient

# loggers
info = lambda msg: sys.stdout.write("info > " + msg + "\n")
warn = lambda msg: sys.stdout.write("warn > " + msg + "\n")

def temp_fix_client():
    token_file = os.environ["TOKEN_FILE"]
    host = os.environ["HOST"]
    port = os.environ["PORT"]
    info(f"Reading token file '{token_file}'")
    with open(token_file) as fp:
        token = fp.read().strip()
    return ContainerClient(token,host=host,port=port)

def find_my_organization_id(client_data):
    id_ = jwt.decode(client_data.token, verify=False)['identity']
    return id_.get('organization_id')

def find_my_node_id(client_data):
    id_ = jwt.decode(client_data.token, verify=False)['identity']
    return id_.get('node_id')