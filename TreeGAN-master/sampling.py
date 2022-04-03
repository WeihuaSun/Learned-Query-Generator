import os
import pickle
import torch
import argparse

import tree_gan
from tree_gan import tree_generator


def sample(args):
    checkpoint = args.ckpt
    if checkpoint != "":
        if os.path.exists(checkpoint):
            java_lang_model_path = checkpoint
        else:
            raise ValueError("can't find checkpoint file:" + checkpoint)
    else:
        java_lang_model_path = os.path.join('models', 'java.model')

    java_data_dir = os.path.join('data', 'java')
    java_bnf_path = os.path.join(java_data_dir, 'java_lang.bnf')
    java_lark_path = os.path.join(java_data_dir, 'java_lang.lark')
    java_text_dir = os.path.join(java_data_dir, 'text_files')
    java_action_getter_path = os.path.join(java_data_dir, 'action_getter.pickle')
    java_action_sequences_dir = os.path.join(java_data_dir, 'action_sequence_files')
    generator_kwargs = {'action_embedding_size': 128}

    with open(java_lang_model_path, 'rb') as f:
        generator_ckp, _ = pickle.load(f)

    a_s_dataset = tree_gan.ActionSequenceDataset(java_bnf_path, java_lark_path, java_text_dir, java_action_getter_path,
                                                 java_action_sequences_dir)

    tree_gen = tree_generator.TreeGenerator(a_s_dataset.action_getter, **generator_kwargs)
    tree_gen.load_state_dict(generator_ckp)

    samples_num = args.n

    with torch.no_grad():
        for _ in range(samples_num):
            _, generated_actions, _, _, _ = tree_gen(max_sequence_length=300)
            print(a_s_dataset.action_getter.construct_text_partial(generated_actions.squeeze().cpu().numpy().tolist()))
            print('---------------------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="number of samples to generate")
    parser.add_argument("--ckpt", type=str, default="", help="pre-trained weights")
    arguments = parser.parse_args()

    sample(arguments)
