import json


def load_function():
    src_path = '../../data/funcom/functions.json'
    with open(src_path, 'r') as fp:
        src = json.load(fp)
    return src


methods_dict = load_function()

methods_lens = [0 for _ in range(57000)]
ind = 0

for method in list(methods_dict.values())[:57000]:
    methods_lens[ind] = len(method)
    ind += 1

print(sum(methods_lens) / 57000)
print(max(methods_lens))
