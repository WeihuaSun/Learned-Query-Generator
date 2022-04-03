import re

with open("/data/java/java_lang.bnf", 'r') as file:
    filedata = file.read()

# filedata = re.sub(r"<([a-zA-Z]+)-([^>]+)>", r'<\1_\2>', filedata)
# filedata = re.sub(r"<([a-z_]+)\s+([a-z_]+)>", r'<\1_\2>', filedata)
# filedata = re.sub(r"<([a-z_]+)\s+([a-z_]+)\s+([a-z_]+)>", r'<\1_\2_\3>', filedata)
# filedata = re.sub(r"<([a-z_]+)\s+([a-z_]+)\s+([a-z_]+)\s+([a-z_]+)>", r'<\1_\2_\3_\4>', filedata)
# filedata = re.sub(r"<([a-z_]+)\s+([a-z_]+)\s+([a-z_]+)\s+([a-z_]+)\s+([a-z_]+)>", r'<\1_\2_\3_\4_\5>', filedata)
# filedata = re.sub(r"<([a-z_]+)\s+([a-z_]+)\s+([a-z_]+)\s+([a-z_]+)\s+([a-z_]+)\s+([a-z_]+)>", r'<\1_\2_\3_\4_\5_\6>', filedata)
# filedata = re.sub(r"<([a-z_]+)\s+([a-z_]+)\s+([a-z_]+)\s+([a-z_]+)\s+([a-z_]+)\s+([a-z_]+)\s+([a-z_]+)>", r'<\1_\2_\3_\4_\5_\6_\7>', filedata)
# filedata = re.sub(r"<([a-z_]+)>", r'\1', filedata)
# filedata = filedata.replace("::=", ":")

filedata = re.sub(r'(?!<[a-z_]+>\s|\s<[a-z_]+>\s|\s::=\s|\s\|\s|\s""\s)(\s)([^\s]+)(?=\s)', r'\1"\2"', filedata)


with open('D:/ml4se/TreeGAN/data/java/formatted.bnf', 'w') as file:
    file.write(filedata)