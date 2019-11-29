import MySQLdb
import pandas as pd

def get_connection():
    conn = MySQLdb.connect(host="localhost", user="root", passwd="tmptArxV", db="joiiup_health", 
                           use_unicode=True, charset="utf8")
    return conn

def do(conn, statement):
    cur = conn.cursor()
    cur.execute(statement)
    return cur

def query(conn, statement):
    cur = do(conn, statement)
    
    field_names = [i[0] for i in cur.description]
    results = cur.fetchall()
    
    df = pd.DataFrame([x for x in results])
    if len(df) == 0:
        return df
    else:
        df.columns = field_names
        return df