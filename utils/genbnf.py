from itertools import combinations

from click import pass_context



class BNFinit():
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
                     ['mc.id', 'mc.movie_id', 'mc.company_id', 'mc.company_type_id', 'mc.note'],
                     ['ci.id', 'ci.person_id', 'ci.movie_id', 'ci.person_role_id', 'ci.note', 'ci.nr_order', 'ci.role_id']
        ]
        self.joinbase = {"t":"t.id","mi":"mi.movie_id","mk":"mk.movie_id","mi_idx":"mi_idx.movie_id","mc":"mc.movie_id","ci":"ci.movie_id"}
    def dump2file(self,path):
        uselist = []
        with open(path,"w") as f:
            f.write("<start> ::= <table_1> | <table_2> | <table_3> | <table_4> | <table_5> | <table_6>\n")
            for i in range(6):#1-6个表
                f.write("<table_{}> ::= ".format(i+1))
                conbin = list(combinations(self.tableset_sn,i+1))
                conbinlist = ["\"["+",".join([ct for ct in cb])+"]\"" for cb in conbin]
                uselist.append(conbinlist)
                conbinlist = [cb+" \"[\" "+" \",\" ".join(["<joinkey_{}>".format("_".join(cb[2:-2].split(","))) for count in range(i)])+" \"]\" \"[\" \"(\" "+(" \")\" \"(\" ".join(["<pred_{}>".format(ce) for ce in cb[2:-2].split(",") ])) + " \")\" \"]\""  for cb in conbinlist]
                f.write("\n | ".join(conbinlist))
                f.write("\n")
            conbinlist = uselist[1]
            for co in conbinlist:
                cs = co[2:-2].split(",")
                f.write("<joinkey_"+"_".join(cs)+"> ::= \""+self.joinbase[cs[0]]+"="+self.joinbase[cs[1]]+"\"\n")
            for i in range(2,6):
                conbinlist = uselist[i]
                for co in conbinlist:
                    cs = co[2:-2].split(",")
                    f.write("<joinkey_"+"_".join(cs)+"> ::= ")
                    csconbin = list(combinations(cs,2))
                    csconbin = ["_".join([ct for ct in cb]) for cb in csconbin]
                    joinkeycob =["<joinkey_{}>".format(cb)for cb in csconbin]
                    f.write(" | ".join(joinkeycob))
                    f.write("\n")
    def bnftolark(self,path,pathout):
        import re
        with open(path, 'r') as file:
            filedata = file.read()
            filedata = re.sub(r"<([a-z_0-9]+)>", r'\1', filedata)
            filedata = filedata.replace("::=", ":")
            filedata = re.sub(r'(\s)([a-z_]+)\s+:\s+""\s+\|\s+([a-z_]+)', r'\1\2 : \3?', filedata)

            with open(path_out, 'w') as file:
                file.write(filedata)    
                    
                    

                    
            
               
    def conbin(self):
        ret = list(combinations(self.tableset,2))
        print(ret)
               
b = BNFinit()
path = "../data/bnftest/temp.bnf"
path_out = "../data/bnftest/temp.lark"
#b.dump2file(path)
b.bnftolark(path,path_out)

