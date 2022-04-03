import os
import pickle
from lark import  Lark
from lark.exceptions import LarkError

import tree_gan.java_parse_utils as parse_utils

java_data_dir = os.path.join('D:/ml4se/TreeGAN/data', 'java')
bnf_path = os.path.join(java_data_dir, 'java_lang.bnf')

texts_dir = os.path.join(java_data_dir, 'text_files')
action_sequences_dir = os.path.join(java_data_dir, 'action_sequence_files')
start = 'start'
lang_grammar_start = 'start'
text_filenames = [dir_entry.name for dir_entry in os.scandir(texts_dir) if dir_entry.is_file()]

action_getter_path = os.path.join(java_data_dir, 'action_getter.pickle')

if os.path.exists(action_getter_path):
    with open(action_getter_path, 'rb') as f:
        action_getter = pickle.load(f)
else:
    my_bnf_parser = parse_utils.CustomBNFParser()
    _, rules_dict, symbol_names = my_bnf_parser.parse_file(bnf_path, start=lang_grammar_start)
    action_getter = parse_utils.SimpleTreeActionGetter(rules_dict, symbol_names)
    if action_getter_path:
        with open(action_getter_path, 'wb') as f:
            pickle.dump(action_getter, f)

lark_path = os.path.join(java_data_dir, 'java_lang.lark')
with open(lark_path) as f:
    parser = Lark(f, keep_all_tokens=True, start=lang_grammar_start)

total_steps = len(text_filenames)
step = 0

print("files num:", total_steps)

for text_filename in text_filenames:
    step += 1
    text_file_path = os.path.join(texts_dir, text_filename)
    text_action_sequence_path = os.path.join(action_sequences_dir, text_filename + '.pickle')
    parsed = True
    if not os.path.exists(text_action_sequence_path):
        with open(text_file_path) as f:
            # Get parse tree of the text file written in the language defined by the given grammar
            try:
                text_tree = parser.parse(f.read(), start=start)
                id_tree = action_getter.simple_tree_to_id_tree(parse_utils.SimpleTree.from_lark_tree(text_tree))
                # Get sequence of actions taken by each non-terminal symbol in 'prefix DFS left-to-right' order
                action_sequences = action_getter.collect_actions(id_tree)
                with open(text_action_sequence_path, 'wb') as f_pickle:
                    pickle.dump(action_sequences, f_pickle)
            except LarkError as e:
                parsed = False
    if not parsed:
        os.remove(text_file_path)
    if step % 500 == 0:
        print(step, "files processed")
