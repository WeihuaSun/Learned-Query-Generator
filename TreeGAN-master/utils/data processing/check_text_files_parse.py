import os
from lark import Lark
import lark.exceptions


java_data_dir = os.path.join('D:/ml4se/TreeGAN/data', 'java')
java_methods_path = os.path.join(java_data_dir, 'text_files')

java_lark_path = os.path.join(java_data_dir, 'java_lang.lark')
with open(java_lark_path) as f:
    parser = Lark(f, start='start')


parsed = 0
not_parsed = 0

methods_f = [os.path.join(java_methods_path, f) for f in os.listdir(java_methods_path)]
step = 2000

for method_f in methods_f[2000:]:
    step += 1
    parsed = False
    with open(method_f, 'r') as f:
        method = f.read()
        try:
            parser.parse(method, start='start')
            parsed += 1
            parsed = True
        except lark.exceptions.LarkError as e:
            print(method)
            print(e)
            print("------------")
            not_parsed += 1
    if not parsed:
        os.remove(method_f)
    if step % 500 == 0:
        print(step)

print(parsed, not_parsed)
