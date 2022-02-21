from fileinput import filename
import sqlite3
from sqlite3 import Error
import os
import shutil

#from argosfeddeep.models import *

#__all__ = ['create_connection','insert_into_table_model','insert_into_table_aggregate','extract_from_table_aggregate','check_database_entries']

#database = r"/mnt/data/argos.db"
#data_path = r'/mnt/data'
#data_path = os.getcwd()

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

def insert_into_table_aggregate(conn,model):
    """
    Create a new project into the projects table
    :param conn:
    :param project:
    :return: project id
    """

    sql = ''' INSERT OR IGNORE INTO aggregate(nodeType,iteration,model_path)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql,model)
    conn.commit()
    return cur.lastrowid

def insert_into_table_nodeModel(conn,model):
    """
    Create a new project into the projects table
    :param conn:
    :param project:
    :return: project id
    """
    sql = ''' INSERT OR IGNORE INTO nodeModel(nodeType,iteration, org_id, training_loss, training_dice, validation_loss, validation_dice, train_model_path)
              VALUES(?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, model)
    conn.commit()
    return cur.lastrowid

def extract_from_table_aggregate(conn,iteration):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    data_path = '/mnt/data'
    path = os.path.join(data_path,'download')

    cur = conn.cursor()
    cur.execute("SELECT * FROM aggregate WHERE iteration=?",(iteration,))
    rows = cur.fetchall()
    for row in rows:
        iteration = row[1]
        filePathNameDatabase = row[2]
    return filePathNameDatabase


def check_database_entries(conn,iteration):
    cur = conn.cursor()
    cur.execute("SELECT * FROM nodeModel WHERE iteration=?",(iteration,))
    rows = cur.fetchall()

    return len(rows)

def flush_database(conn):
    cur = conn.cursor()
    cur.execute("DELETE from nodeModel;",)
    cur.execute("DELETE from aggregate;",)

