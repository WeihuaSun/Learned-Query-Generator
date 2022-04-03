import re

with open("./data/sql/SQL_sample.bnf", 'r') as file:
    filedata = file.read()

filedata = re.sub(r"<([a-z_]+)>", r'\1', filedata)
filedata = filedata.replace("::=", ":")
filedata = re.sub(r'(\s)([a-z_]+)\s+:\s+""\s+\|\s+([a-z_]+)', r'\1\2 : \3?', filedata)

with open('./data/sql/sql_lang.lark', 'w') as file:
    file.write(filedata)