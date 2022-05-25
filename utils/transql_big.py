from collections import OrderedDict
from ntpath import join
import re
import os
from typing import Dict, NamedTuple, Optional, Tuple, Any


class ParserSql():

    def __init__(self):
        self.tableset = ['title','movie_info','movie_keyword','movie_info_idx','movie_companies','cast_info']
        self.tableset_sn = ['t','mi','mk','mi_idx','mc','ci']
        self.colset = ['t.id', 't.title', 't.imdb_index', 't.kind_id', 't.production_year', 't.imdb_id', 't.phonetic_code', 't.episode_of_id', 't.season_nr', 't.episode_nr', 't.series_years', 't.md5sum',
                       'mi.id', 'mi.movie_id', 'mi.info_type_id', 'mi.info', 'mi.note', 
                       'mk.id', 'mk.movie_id', 'mk.keyword_id', 
                       'mi_idx.id', 'mi_idx.movie_id', 'mi_idx.info_type_id', 'mi_idx.info', 'mi_idx.note',
                       'mc.id', 'mc.movie_id', 'mc.company_id', 'mc.company_type_id', 'mc.note', 
                       'ci.id', 'ci.person_id', 'ci.movie_id', 'ci.person_role_id', 'ci.note', 'ci.nr_order', 'ci.role_id']
        self.cols = [['t.id', 't.title', 't.imdb_index', 't.kind_id', 't.production_year', 't.imdb_id', 't.phonetic_code', 't.episode_of_id', 't.season_nr', 't.episode_nr', 't.series_years', 't.md5sum'],
                     ['mi.id', 'mi.movie_id', 'mi.info_type_id', 'mi.info', 'mi.note'],
                     ['mk.id', 'mk.movie_id', 'mk.keyword_id'],
                     ['mi_idx.id', 'mi_idx.movie_id', 'mi_idx.info_type_id', 'mi_idx.info', 'mi_idx.note'],
                     [ 'mc.id', 'mc.movie_id', 'mc.company_id', 'mc.company_type_id', 'mc.note'],
                     ['ci.id', 'ci.person_id', 'ci.movie_id', 'ci.person_role_id', 'ci.note', 'ci.nr_order', 'ci.role_id']
        ]
        self.samplefromsql = OrderedDict([(c,set([]))for c in self.colset])
    def parsertable(self,sql):
            reg= r"(FROM\s)(([a-zA-Z_]+\s[a-zA-Z_]+\,?)+)(\sWHERE)"
            vm = re.search(reg,sql)
            jointables = vm.group(2).split(",")
            return jointables
    def parser(self,srcpath,outpath):
        with open(srcpath,"r") as f:
            index = 0
            for sql in f:
                index = index + 1
                tables = self.parsertable(sql)
                tables_sn = [t.split(" ")[1] for t in tables]
                tables_sn.sort(key=lambda x : self.tableset_sn.index(str(x)))
                joinkey = r"([a-zA-Z_]+\.[a-zA-Z_]+)\=([a-zA-Z_]+\.[a-zA-Z_]+)"
                pred = r"([a-zA-Z_]+\.[a-zA-Z_]+)(=|!=|<=|>=|>|<)([0-9.]+|\'[a-zA-Z_]+\')"
                joinkeys = [[m.group(1),m.group(2)] for m in re.finditer(joinkey, sql)]
                joinkeys = [sorted(jk,key = lambda x: self.colset.index(x)) for jk in joinkeys]
                joinkeys = ["{}={}".format(jk[0],jk[1]) for jk in joinkeys]
                preds = [[m.group(1),m.group(2),m.group(3)] for m in re.finditer(pred, sql)]
                preds.sort(key=lambda x: self.colset.index(x[0]))
                predsdict = OrderedDict([(tsn,[])for tsn in tables_sn])
                for prd in preds:
                    predsdict[prd[0].split(".")[0]].append("{}{}{}".format(prd[0],prd[1],prd[2]))
                self.dump2file(tables_sn,joinkeys,predsdict,outpath,index)
                
    def dump2file(self,tables_sn,joinkeys,predsdict,data_processed,index):
        with open(os.path.join(data_processed,f"job_{index}"),"w") as f:
            sql = "["+",".join(tables_sn)+"]["
            sql = sql+",".join(joinkeys) +"]["
            index = index +1
            for key in predsdict:
                sql = sql + "(" + ",".join(predsdict[key]) + ")"
            sql = sql + "]"
            f.write(sql)    
    def resql(self,srcpath,outpath):
        path_list = os.listdir(srcpath)
        index = 0
        reg = r"\[(.*)\]\[(.*)\]\[(.*)\]"
        pred = r"([a-zA-Z_]+\.[a-zA-Z_]+)(=|!=|<=|>=|>|<)([0-9.]+|\'[a-zA-Z_]+\')"
        for path in path_list:
            index = index+1
            with open(os.path.join(srcpath,path),"r") as file:
                psql = file.readline() 
                vm = re.search(reg,psql)
                tables = vm.group(1).split(",")
                tables = ["{} {}".format(self.tableset[self.tableset_sn.index(t)],t)  for t in tables]
                joins = vm.group(2).split(",")
                preds = vm.group(3)
                preds = [m.group(0) for m in re.finditer(pred,preds)]
                with open(os.path.join(outpath,f"job{index}"),"w")as f:
                    sql = "SELECT COUNT(*) FROM "
                    sql = sql + ",".join(tables)
                    sql = sql+ " WHERE "
                    sql = sql+ " AND ".join(joins+preds)
                    sql = sql+ ";"
                    f.write(sql)
                        
class Query(NamedTuple):
    predicates: Dict[str, Optional[Tuple[str, Any]]]
    ncols: int
def new_query(columns,ncols):
    return Query(predicates=OrderedDict.fromkeys(columns, None),
                 ncols=ncols)    
path_process = "..\data\job-big\processed"
path_rebuild = "..\data\job-big\\rebuild"
datapath = "..\data\job-big\job-light.sql"
#p = ParserSql()
#p.parser(datapath,path_process)
#p.resql(path_process,path_rebuild)