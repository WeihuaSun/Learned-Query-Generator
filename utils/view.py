"""
SQL VIEW
"""

import sys
import os
import re
import psycopg2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from database.dataset import Postgres,Database
#from ..Constants import DATABASE_URL



path_percol = "..\data\job-big\\rebuild"
path_percol_out = "..\data\job-big\\rebuild"
path_sample = "..\data\job-big\\rebuild"
path_sample_out = "..\output\multi\sample\\rebuild"


class Show:
    def __init__(self,sample_path,sample_outpath,syntax_path,syntax_outpath):
        self.sample_path = sample_path
        self.sample_outpath = sample_outpath
        self.syntax_path = syntax_path
        self.syntax_outpath = syntax_outpath
        #self.databse = Database()
        self.colset = ['t.id', 't.title', 't.imdb_index', 't.kind_id', 't.production_year', 't.imdb_id', 't.phonetic_code', 't.episode_of_id', 't.season_nr', 't.episode_nr', 't.series_years', 't.md5sum',
                       'mi.id', 'mi.movie_id', 'mi.info_type_id', 'mi.info', 'mi.note', 
                       'mk.id', 'mk.movie_id', 'mk.keyword_id', 
                       'mi_idx.id', 'mi_idx.movie_id', 'mi_idx.info_type_id', 'mi_idx.info', 'mi_idx.note',
                       'mc.id', 'mc.movie_id', 'mc.company_id', 'mc.company_type_id', 'mc.note', 
                       'ci.id', 'ci.person_id', 'ci.movie_id', 'ci.person_role_id', 'ci.note', 'ci.nr_order', 'ci.role_id']
    
    def colheat(self):
        """_summary_
            列分布热图
        """
        self.getPearson(self.sample_path,"样本")
        self.getPearson(self.sample_outpath,"抽样方法") 
        self.getPearson(self.syntax_outpath,"语法方法")       
    def getPearson(self,root,choice=""):
        """获得相关系数

        Args:
            root (_type_): _description_
            choice (str, optional): 方法. Defaults to "".
        """
        todata = {a:[] for a in self.colset}
        path_list = os.listdir(root)
        for path in path_list:
            with open(os.path.join(root,path),"r") as file:  
                sql = file.readline()
                for c in self.colset:
                    index = [m.start() for m in re.finditer(c, sql)]
                    todata[c].append(len(index))
        adata=[]
        for k in todata.keys():
            adata.append(todata[k])
        pdata = pd.DataFrame(adata,index=self.colset,columns=[i for i in range(70)]).T
        cm = pdata.corr()
        sns.set(font_scale=1.25)
        hm = sns.heatmap(cm,
                 cbar=True,
                 annot=False,
                 square=True,
                 fmt=".3f",
                 vmin=0,             #刻度阈值
                 vmax=1,
                 linewidths=.5,
                 cmap="RdPu",        #刻度颜色
                 annot_kws={"size": 10},
                 xticklabels=False,
                 yticklabels=False)             #seaborn.heatmap相关属性
            # 解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # plt.ylabel(fontsize=15,)
        # plt.xlabel(fontsize=15)
        plt.title(f"列相关系数---{choice}", fontsize=20)
        plt.show()
         
    def showcolnum(self):
        """显示各个列的数量
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        sample_data = self.checkcolnum(self.sample_path,self.colset)
        sample_gendata =self.checkcolnum(self.sample_outpath,self.colset)
        syntax_data = self.checkcolnum(self.syntax_path,self.colset)
        syntax_gendata = self.checkcolnum(self.syntax_outpath,self.colset)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.bar(x=self.colset,height=sample_data,label="训练样本数据",alpha=0.5)
        ax.bar(x=self.colset,height =sample_gendata, label="TreeGan生成数据",alpha=0.5)        
        ax.set_title("列出现次数--抽样方法")
        ax.legend()   
        plt.show() 
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.bar(x=self.colset,height=syntax_data,label="训练样本数据",alpha=0.5)
        ax.bar(x=self.colset,height =syntax_gendata, label="TreeGan生成数据",alpha=0.5)
        ax.set_title("列出现次数--语法方法")
        ax.legend()   
        plt.show()   
    def checkcolnum(self,root,collist):
        import re
        value = [0 for i in range(len(collist))]
        #value = []
        path_list = os.listdir(root)
        for path in path_list:
            with open(os.path.join(root,path),"r") as file:  
                sql = file.readline()
                for i in range(len(collist)):
                    index = [m.start() for m in re.finditer(collist[i], sql)]
                    value[i] = value[i] + len(index) 
        return value

    def show_predlength(self,token = 'AND'):
        """谓词长度

        Args:
            token (str, optional): 连接词. Defaults to 'AND'.
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        sample_data = self.checkSyntax(self.sample_path,token)
        print("sample:"+self.sample_path)
        sample_gendata =self.checkSyntax(self.sample_outpath,token)
        print("sample_gen:",self.sample_outpath)
        syntax_data = self.checkSyntax(self.syntax_path,token)
        print("syntax:",self.syntax_path)
        syntax_gendata = self.checkSyntax(self.syntax_outpath,token)
        print("syntax_gen:",self.syntax_outpath)
        kwargs = {
        "bins": 20,
        "histtype": "bar",
        "alpha": 0.5
        }
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.hist(x=sample_data, label="训练样本数据", **kwargs)
        ax.hist(sample_gendata, label="TreeGan生成数据", **kwargs)
        ax.set_title("抽样方法")
        ax.legend()   
        plt.show() 
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.hist(x=syntax_data, label="训练样本数据", **kwargs)
        ax.hist(x=syntax_gendata, label="TreeGan生成数据", **kwargs)
        ax.set_title("语法方法")
        ax.legend()   
        plt.show()  
    def checkSyntax(self,root,token): 
        path_list = os.listdir(root)
        countlist = []
        for path in path_list:
            with open(os.path.join(root,path),"r") as file: 
                sql = file.readline()
                countlist.append(sql.count(token))
            file.close()
        return countlist
    def showErr(self):
        types = ['normal','unmeaning','repeat']
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        sample_data = self.checkErr(self.sample_path)
        sample_gendata =self.checkErr(self.sample_outpath)
        syntax_data = self.checkErr(self.syntax_path)
        syntax_gendata = self.checkErr(self.syntax_outpath)
        plt.figure()
        plt.subplot(221)
        plt.pie(x=sample_data,labels =types ,autopct='%.3f%%')
        plt.title("抽样样本")
        plt.subplot(222)
        plt.pie(x=sample_gendata,labels =types ,autopct='%.3f%%')
        plt.title("抽样生成")
        plt.subplot(223)
        plt.pie(x=syntax_data,labels =types ,autopct='%.3f%%')
        plt.title("语法样本")
        plt.subplot(224)
        plt.pie(x=syntax_gendata,labels =types ,autopct='%.3f%%')
        plt.title("语法生成")
        plt.show()
        return

    def checkcols(self,sql):
        #检查无意义
        #1.对于非数值列
        #   COL == 'A'  AND  COL !='A'无意义
        #   COL == 'B'  AND  COL !='B' 无意义
        #2.对于数值列
        #   COL <= MIN-
        #   COL => MAX+
        #   COL=< A AND COL>=B 且 A<B
        #   COL == 'A' AND COL>A OR COL <A
        #   同时也包括非数值列的相同类型
        #检查重复
        # COL != 'A'  AND  COL !='A'重复
        # # COL == 'A'  AND  COL =='A'重复
        # COL>=A AND COL>=B
        # COL<=A AND COL<=B
        # COL = A AND COL<=B  A IN B field
        for c in self.colset:
            index = [m.start() for m in re.finditer(c, sql)]
            if len(index)>0:
                col = self.table.columns[c]
                reg = r"\d+(\.)?\d*"
                oplist = []
                valuelist = []
                for i in index:
                    oplist.append(sql[i+len(c)+1:i+len(c)+3])
                    vm = re.search(reg,sql[i+len(c)+3:])
                    valuelist.append(vm.group(0))
                preop = None
                yval = None
                nval = []
                aboveval =None
                belowval = None
                valuelist = [float(v) for v in valuelist]
                for (op,val) in zip(oplist,valuelist):
                        if preop is not None:
                            if op =='<=':
                                if val<col.minval: #COL <= MIN-
                                    return 'unmeaning'
                                if yval is not None and val<yval:# COL == 'A' AND  COL <=A-
                                    return 'unmeaning'
                                if belowval is not None and belowval>val:  #COL=< A AND COL>=B 且 A<B
                                    return 'unmeaning'
                                if aboveval is not None: # COL<=A AND COL<=B
                                    return 'repeat'
                                if yval is not None:#COL = A AND COL<=B  A IN B field
                                    return 'repeat'
                                if len(nval)>0:
                                    for n in nval:
                                        if n >val:
                                            return 'repeat'
                                aboveval = val
                                preop = op
                            elif op =='>=':
                                if val>col.maxval:
                                    return 'unmeaning'  
                                if yval is not None and val>yval:
                                    return 'unmeaning'
                                if aboveval is not None and aboveval<val:
                                    return 'unmeaning'
                                if belowval is not None:
                                    return 'repeat'
                                if yval is not None:
                                    return 'repeat'
                                if len(nval)>0:
                                    for n in nval:
                                        if n <val:
                                            return 'repeat'
                                belowval = val
                                preop = op 
                            if op =='==' :
                                if yval!=val: #  COL == 'B'  AND  COL =='C'无意义
                                    return 'unmeaning'
                                if belowval is not None and belowval>val:
                                    return 'unmeaning'
                                if aboveval is not None and aboveval<val:
                                    return 'unmeaning'
                                if aboveval is not None:
                                    return 'repeat'
                                if belowval is not None:
                                    return 'repeat'
                            elif op =='==' and val in nval: #COL == 'A'  AND  COL !='A'无意义
                                return 'unmeaning'
                            elif op =='!=' and yval == val: #COL !='A' AND COL == 'A'无意义
                                return 'unmeaning'
                            elif op =='!=' and val in nval: # COL != 'A'  AND  COL !='A'重复
                                return 'repeat'
                            elif op =='!=':
                                if aboveval is not None and val >aboveval:
                                    return 'repeat'
                                if belowval is not None and val <belowval:
                                    return 'repeat'
                            elif op =='==' and yval==val: # COL == 'A'  AND  COL =='A'重复
                                return 'repeat'
                            elif op =='!=':
                                preop = op
                                nval.append(val)
                            elif op =='==':
                                preop = op
                                yval =val
                        else:
                            preop = op
        return 'normal'        

    def checkErr(self,root):
        normal = 0
        unmeaning = 0
        repeat = 0
        path_list = os.listdir(root)
        for path in path_list:
            with open(os.path.join(root,path),"r") as file: 
                sql = file.readline()
                if self.checkcols(sql)=='unmeaning':
                    unmeaning = unmeaning+1
                elif self.checkcols(sql)=='repeat':
                    repeat = repeat+1
            file.close()
        normal = len(path_list)-unmeaning-repeat
        return [normal,unmeaning,repeat]

    def show_indatabase(self,choice="card"):
        sample_data = self.checkdata(self.sample_path,choice)
        sample_gendata = self.checkdata(self.sample_outpath,choice)
        syntax_data = self.checkdata(self.syntax_path,choice)
        syntax_gendata = self.checkdata(self.syntax_outpath,choice)
        print(syntax_data)
        data = pd.DataFrame({
            "sample_data":sample_data,
            "sample_gendata": sample_gendata,
            "syntax_data":syntax_data,
            "syntax_gendata":syntax_gendata
        })
        data.boxplot()
        plt.title(choice)
        plt.ylabel("10^n")
        plt.show()
    def checkdata(self,root,choice='cost'):
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        cur = conn.cursor()
        result = []
        path_list = os.listdir(root)
        for path in path_list:
            with open(os.path.join(root,path),"r") as file: 
                sql = file.readline()
                if(choice == 'cost'):
                    sql =sql.replace("SELECT",f"EXPLAIN SELECT")
                    cur.execute(sql)
                    rows = cur.fetchall() 
                    start = rows[0][0].index("cost")+5
                    end = rows[0][0].index("..")
                    result.append(np.log10(float(rows[0][0][start:end])))
                if(choice == 'card'):
                    cur.execute(sql)
                    rows = cur.fetchall()
                    if(rows[0][0]==0):
                        result.append(0)
                    else:    
                        result.append((rows[0][0]))
            file.close()
        cur.close()
        conn.close()
        return result

class showloss:
    def __init__(self):
        
        gen_losses = []
        with open('../train_history/5-9-3/output/multi/sample/stats/generator_rewards', 'rb') as f:
            for i in range(20):
                gen_loss = pickle.load(f)
                gen_losses.extend(gen_loss)
            self.make_rewards_plot(gen_losses)
        gen_losses = []
        with open('../train_history/5-9-3/output/multi/sample/stats/generator_losses', 'rb') as f:  
            for i in range(20):
                gen_loss = pickle.load(f)
                gen_losses.extend(gen_loss)
                self.make_losses_plot(gen_loss,"gen")
            self.make_losses_plot(gen_losses,"gen")
        gen_losses = []
        with open('../train_history/5-9-3/output/multi/sample/stats/discriminator_losses', 'rb') as f:  
            for i in range(20):
                gen_loss = pickle.load(f)
                gen_losses.extend(gen_loss)
            self.make_losses_plot(gen_losses,"d_loss")
        
    def moving_average(self,a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        ret[n:] = ret[n:] / n
        ret[:n] = ret[:n] / np.arange(1, n + 1)
        return ret
#autoencoder 稀疏数据
#编码，分布
#训练集，增加数据集
#标签
#加深，加宽网络，减小，变浅
#正则化标准
#WGAN-GP
#验证集
#网格搜索
    def make_losses_plot(self,losses, title):
        flat_losses = [item for sublist in losses for item in sublist]
        plt.plot(flat_losses, label='loss')
        plt.plot(self.moving_average(np.array(flat_losses),5), label='moving avg')
        plt.title(title)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        x_ids = []
        last = 0
        for l in losses:
            last += len(l)
            x_ids.append(last)
        plt.xticks(x_ids, [i + 1 for i in range(len(losses))])
        plt.legend()
        plt.show()
    def make_loss_plot(self,loss, title):
        plt.plot(loss, label='loss')
        #plt.plot(self.moving_average(np.array(loss), 3), label='moving avg')
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.title(title)
        plt.legend()
        plt.show()

    def make_rewards_plot(self,rewards):
        plt.plot(rewards, label='avg episode reward')
        #plt.plot(self.moving_average(np.array(gen_rewards), 50), label='moving avg')
        plt.xlabel('episode')
        plt.ylabel('avg reward') 
        plt.title("generator rewards")
        plt.legend()
        plt.show()






s = Show(path_sample,path_sample_out,path_percol,path_percol_out)
#s.show_indatabase("card")
#s.show_indatabase("cost")
s.colheat()
s.show_predlength()
s.showcolnum()
#s.showErr()
sl = showloss()
