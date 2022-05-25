import os
import psycopg2
import pickle
import sys
from collections import OrderedDict
from constants import DATABASE_URL
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
gettablename = "SELECT tablename FROM pg_tables WHERE tablename NOT LIKE 'pg%' AND tablename NOT LIKE 'sql_%'"
gettableinfo = "SELECT tablename,obj_description(relfilenode,'pg_class') FROM pg_tables a, pg_class b WHERE a.tablename = b.relname AND a.tablename NOT LIKE 'pg%' AND a.tablename NOT LIKE 'sql_%' ORDER BY  a.tablename;"
getcols = "SELECT col_description(a.attrelid, a.attnum) AS comment,format_type(a.atttypid,a.atttypmod) AS type,a.attname AS name,a.attnotnull AS notnull FROM pg_class AS c,pg_attribute AS a WHERE c.relname = \'{}\'  AND a.attrelid = c.oid AND a.attnum>0 "

class Postgres():
    def __init__(self):
        self.conn = psycopg2.connect(DATABASE_URL)
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()
    def query_sql(self,sql):
        self.cursor.execute(sql)
        res = self.cursor.fetchall()
        return res


    def query_sql_exp(self, sql):
        sql = 'explain(format json) {}'.format(sql)
        self.cursor.execute(sql)
        res = self.cursor.fetchall()
        card = res[0][0][0]['Plan']['Plan Rows']
        return card

    def getinfo(self):
        tables = self.query_sql(gettablename)
        tables = [t[0] for t in tables]
        table_cols =OrderedDict([(t,self.query_sql(getcols.format(t)))for t in tables])
        return tables,table_cols


class Database():
    def __init__(self):
        self.post =  Postgres()
        self.get_tables()
    def get_tables(self):
        self.tablename,self.colinfo = self.post.getinfo()
        self.getinfo()
    def dumpinfo(self,dbinfo):
        with open("../data/temp/dbinfo.pkl","wb") as f:
            pickle.dump(dbinfo,f)
    def getinfo(self):
        if os.path.exists("../data/temp/dbinfo.pkl"):
            with open("../data/temp/dbinfo.pkl","rb") as f:
                db = pickle.load(f)
            self.tables = db
        else:
            self.tables =  OrderedDict([(t, Table(t,self.colinfo[t])) for t in self.tablename])
            self.dumpinfo(self.tables)
    def sample(self,table,percentage=0.001):
        return self.post.query_sql("select* from {} TABLESAMPLE bernoulli({});".format(table,percentage))
        

           
class Table():
    def __init__(self,name,colinfo):
        self.name = name
        self.columns = OrderedDict([(col[2], Column(col,self.name)) for col in colinfo])
        self.colnum = len(self.columns)
        
class Column():
    def __init__(self,info,table):
        post = Postgres() 
        self.name = info[2]
        self.dtype = info[1]
        self.notnull = info[3]
        self.maxval = post.query_sql("SELECT MAX({}) FROM {};".format(self.name,table))[0][0]
        self.minval = post.query_sql("SELECT MIN({}) FROM {};".format(self.name,table))[0][0]
