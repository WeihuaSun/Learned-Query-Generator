import os
import pickle
import argparse
import time
import torch
import TreeGan
from Constants import DATA_ROOT,MODEL_ROOT,OUTPUT_ROOT
from TreeGan.learning_utils import tree_gan_evaluate,tree_generator
from utils.transql_big import ParserSql
torch.set_default_dtype(torch.float32)

def train(params):
    mean_reward, (tree_gen, tree_dis), episode_reward_lists = tree_gan_evaluate(**params)
def test(model_path,a_s_dataset,gennum,outpath,repath,parser:ParserSql):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            generator_ckp, _ = pickle.load(f)
        tree_gen = tree_generator.TreeGenerator(a_s_dataset.action_getter, **generator_kwargs)
        tree_gen.load_state_dict(generator_ckp)
        with torch.no_grad():
            for i in range(gennum):
                _, generated_actions, _, _, _ = tree_gen(max_sequence_length=500)
                with open(f"{outpath}\query_{i}.sql", 'w',newline="") as f:
                    sql = a_s_dataset.action_getter.construct_text_partial(generated_actions.squeeze().cpu().numpy().tolist())
                    f.writelines(sql)
                f.close()
        parser.resql(outpath,repath)


if __name__ == '__main__':
    START = time.process_time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', action='store_true')
    parser.set_defaults(train = False)
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(test=False)
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help="train from pretrained weights")
    parser.set_defaults(pretrained=False)
    parser.add_argument("--choice", type=str, default="multi", help="multiple tables or single tables")
    parser.add_argument("--method",type=str, default="sample",help = "methods to generate concrete values")
    parser.add_argument("--gennum",type=int, default=70)
    arguments = parser.parse_args()
    data_dir = os.path.join(DATA_ROOT,"{}-table\{}".format(arguments.choice,arguments.method))
    out_dir =  os.path.join(OUTPUT_ROOT,"{}\{}".format(arguments.choice,arguments.method))
    model_path = os.path.join(MODEL_ROOT,"{}_{}.model".format(arguments.choice,arguments.method))
    bnf_path = os.path.join(data_dir,"sql.bnf")
    lark_path = os.path.join(data_dir,"sql_lang.lark")
    stats_dir = os.path.join(out_dir,"stats")
    text_path = os.path.join(DATA_ROOT,"job-light\job-light.sql")
    text_dir = os.path.join(DATA_ROOT,"job-light\job-light-process")
    action_getter_path = os.path.join(data_dir,"action_getter.pickle")
    action_sequences_dir = os.path.join(data_dir,"actsqu")
    parser = ParserSql()
    #parser.parser(text_path,text_dir)
    generator_kwargs = {'action_embedding_size': 128}
    a_s_dataset=TreeGan.ActionSequenceDataset(bnf_path, lark_path, text_dir,
                                                    action_getter_path, action_sequences_dir)
    if arguments.test:
        test_params = dict(
            model_path=model_path,
            a_s_dataset=a_s_dataset,
            gennum = arguments.gennum,
            outpath = os.path.join(out_dir,"gen"),
            repath = os.path.join(out_dir,"rebuild"),
            parser = parser
        )
        test(**test_params)
    else:
        if arguments.pretrained:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    generator_ckp, discriminator_ckp = pickle.load(f)
        else:
            generator_ckp, discriminator_ckp = None, None
        all_params = dict(
            sql_lang_model_path=model_path,
            stats_save_path=stats_dir,
            a_s_dataset=a_s_dataset,
            generator_ckp=generator_ckp,
            discriminator_ckp=discriminator_ckp,
            generator_kwargs={'action_embedding_size': 128},
            discriminator_kwargs={'action_embedding_size': 128},
            num_data_loader_workers=1,
            max_total_step=100000,  # min number of steps to take during generator training#!!20
            initial_episode_timesteps=150,  # initial max time steps in one episode
            final_episode_timesteps=300,  # final max time steps in one episode (MUST NOT EXCEED 'buffer_timestep')
            episode_timesteps_log_order=0,
            gamma=0.99,  # discount factor
            gae_lambda=0.95,  # lambda value for td(lambda) returns
            eps_clip=0.2,  # clip parameter for PPO
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            random_seed=1234,
            lr=1e-4,#?
            buffer_timestep=10000,
            lr_decay_order=5,
            k_epochs=5,
            buffer_to_batch_ratio=2,
            optimizer_betas=(0.5, 0.75),
            # PRE-TRAINING HYPER PARAMETERS
            pre_train_epochs=100,
            pre_train_batch_size=64,
            # DISCRIMINATOR TRAINING HYPER PARAMETERS
            discriminator_train_epochs=5,
            discriminator_train_batch_size=64,
            # GAN TRAINING HYPER PARAMETERS
            gan_epochs=20
        )  
        train(params=all_params)  
    print('ELAPSED TIME (sec): ' + str(time.process_time() - START))
    print('---------------------------------')
    
    