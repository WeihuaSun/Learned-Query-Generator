import os
import pickle
import torch
import argparse
import csv
import tree_gan
from tree_gan import tree_generator


def sample(args):
    checkpoint = args.ckpt
    if checkpoint != "":
        if os.path.exists(checkpoint):
            sql_lang_model_path = checkpoint
        else:
            raise ValueError("can't find checkpoint file:" + checkpoint)
    else:
        sql_lang_model_path = os.path.join('models', f'job_{args.choice}.model')

    sql_data_dir = os.path.join('data', 'multi-table')
    sql_bnf_path = os.path.join(sql_data_dir, f'sql_{args.choice}.bnf')
    sql_lark_path = os.path.join(sql_data_dir, f'sql_lang_{args.choice}.lark')
    sql_text_dir = os.path.join(sql_data_dir, f'job-light')
    sql_action_getter_path = os.path.join(sql_data_dir, f'action_getter_{args.choice}.pickle')
    sql_action_sequences_dir = os.path.join(sql_data_dir, f'actsqu_{args.choice}')
    generator_kwargs = {'action_embedding_size': 128}

    with open(sql_lang_model_path, 'rb') as f:
        generator_ckp, _ = pickle.load(f)

    a_s_dataset = tree_gan.ActionSequenceDataset(sql_bnf_path, sql_lark_path, sql_text_dir, sql_action_getter_path,
                                                 sql_action_sequences_dir)

    tree_gen = tree_generator.TreeGenerator(a_s_dataset.action_getter, **generator_kwargs)
    tree_gen.load_state_dict(generator_ckp)

    samples_num = args.n

    with torch.no_grad():
        for i in range(samples_num):
            _, generated_actions, _, _, _ = tree_gen(max_sequence_length=500)
            with open(f"./output/{args.choice}/query{i}.sql", 'w',newline="") as f:
                sql = a_s_dataset.action_getter.construct_text_partial(generated_actions.squeeze().cpu().numpy().tolist())
                f.writelines(sql)
            f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=70, help="number of samples to generate")
    parser.add_argument("--ckpt", type=str, default="", help="pre-trained weights")
    parser.add_argument("--choice", type=str, default="sample", help="pre-trained weights")
    arguments = parser.parse_args()

    sample(arguments)

   
            
