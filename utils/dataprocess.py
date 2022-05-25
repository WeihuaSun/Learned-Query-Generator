from collections import OrderedDict
import re
import os
from typing import Dict, NamedTuple, Optional, Tuple, Any
from Constants import DATA_ROOT

class ParserSql():
    """sql语句的转换和恢复
    """
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
    def parser(self,srcpath,outpath):
        with open(srcpath,"r") as f:
            index = 0
            for sql in f:
                index = index + 1
                joinkey = r"([a-zA-Z_]+)\.([a-zA-Z_]+)\=([a-zA-Z_]+)\.([a-zA-Z_]+)"
                pred = r"([a-zA-Z_]+)\.([a-zA-Z_]+)(=|!=|<=|>=|>|<)([0-9.]+|\'[a-zA-Z_]+\')"
                joinkeys = [[m.group(1),m.group(2),m.group(3),m.group(4)] for m in re.finditer(joinkey, sql)]
                preds = [[m.group(1),m.group(2),m.group(3),m.group(4)] for m in re.finditer(pred, sql)]
                rankpreds = new_query(self.colset,len(preds))  
                tables = set([])
                for join in joinkeys:
                    tables.add(join[0])
                    tables.add(join[2])
                jointables = OrderedDict([(t,[])for t in tables])
                for pred in preds:
                    rankpreds.predicates[f"{pred[0]}.{pred[1]}"] = (pred[2],pred[3])
                    self.samplefromsql[f"{pred[0]}.{pred[1]}"].add(pred[3])
                for col, pred in rankpreds.predicates.items():
                    if pred is None:
                        continue 
                    op, val = pred
                    jointables[col.split(".")[0]].append(f"{col}{op}{val}")
                self.dump2file(joinkeys,jointables,index,outpath)
                
    def dump2file(self,joinkeys,jointables,index,data_processed):
        with open(os.path.join(data_processed,f"job_{index}"),"w") as f:
            sql = ""
            for join in joinkeys:
                    sql = sql + f"({join[0]}.{join[1]}["+",".join(jointables[join[0]])+"],"+f"{join[2]}.{join[3]}["+",".join(jointables[join[2]])+"])"
            f.write(sql)    
    def resql(self,srcpath,outpath):
        path_list = os.listdir(srcpath)
        index = 0
        reg_join = r"\(([a-zA-Z_]+)\.([a-zA-Z_]+)\[([^\]]*)\]\,([a-zA-Z_]+)\.([a-zA-Z_]+)\[([^\]]*)\]\)"
        for path in path_list:
            index = index+1
            with open(os.path.join(srcpath,path),"r") as file:
                tables = set([])
                psql = file.readline() 
                preds = []
                for m in re.finditer(reg_join, psql):
                    tables.add(m.group(1))
                    tables.add(m.group(4))
                    preds.append("{}.{}={}.{}".format(m.group(1),m.group(2),m.group(4),m.group(5)))
                    reg_pred = r"([a-zA-Z_]+)\.([a-zA-Z_]+)(=|!=|<=|>=|>|<)([0-9.]+|\'[a-zA-Z_]+\')"
                    for vm in re.finditer(reg_pred,m.group(3)):
                        preds.append(vm.group(0))
                    for vm in re.finditer(reg_pred,m.group(6)):
                        preds.append(vm.group(0))
                with open(os.path.join(outpath,f"job{index}"),"w")as f:
                    sql = "SELECT COUNT(*) FROM "
                    ta = tables.pop()
                    sql = sql + "{} {}".format(self.tableset[self.tableset_sn.index(ta)],ta)
                    for t in tables:
                         sql = sql + ",{} {}".format(self.tableset[self.tableset_sn.index(t)],t)
                    sql = sql+ " WHERE "
                    sql = sql+ " AND ".join(preds)
                    sql = sql+ ";"
                    f.write(sql)
                        
class Query(NamedTuple):
    predicates: Dict[str, Optional[Tuple[str, Any]]]
    ncols: int
def new_query(columns,ncols):
    return Query(predicates=OrderedDict.fromkeys(columns, None),
                 ncols=ncols)    
