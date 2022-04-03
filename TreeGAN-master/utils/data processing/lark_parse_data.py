import json
import os
from lark import Lark
import lark.exceptions
import re

def load_function():
    src_path = '../../data/funcom/functions.json'
    with open(src_path, 'r') as fp:
        src = json.load(fp)
    return src


methods_dict = load_function()

java_data_dir = os.path.join('D:/ml4se/TreeGAN/data', 'java')
java_lark_path = os.path.join(java_data_dir, 'java_lang.lark')

with open(java_lark_path) as f:
    parser = Lark(f, start='method_declaration')

parsed = 0
not_parsed = 0

java_methods_path = os.path.join(java_data_dir, 'text_files')

for (method_id, method) in list(methods_dict.items())[:100000]:
    method = re.sub(r'//[^\n]+', r'', method)
    method = re.sub(r'[\t\r\n]+', r' ', method)
    method = re.sub(r' +', r' ', method)
    try:
        parser.parse(method, start='method_declaration')

        cur_path = os.path.join(java_methods_path, "method" + str(method_id) + ".txt")
        if not(os.path.exists(cur_path)):
            with open(cur_path, 'w') as f:
                f.write(method)

        parsed += 1
    except lark.exceptions.LarkError as e:
        # print(method)
        # print("------------")
        not_parsed += 1

print("parsed", parsed, "not parsed", not_parsed)