import os
import pickle

java_data_dir = os.path.join('D:/ml4se/TreeGAN/data', 'java')
texts_dir = os.path.join(java_data_dir, 'text_files')
action_sequences_dir = os.path.join(java_data_dir, 'action_sequence_files')
text_filenames = [dir_entry.name for dir_entry in os.scandir(texts_dir) if dir_entry.is_file()]

removed = 0

for text_filename in text_filenames:
    text_file_path = os.path.join(texts_dir, text_filename)
    text_action_sequence_path = os.path.join(action_sequences_dir, text_filename + '.pickle')
    with open(text_action_sequence_path, 'rb') as f:
        actions, _ = pickle.load(f)
    if len(actions) > 250:
        os.remove(text_file_path)
        os.remove(text_action_sequence_path)
        removed += 1

print("removed", removed)
