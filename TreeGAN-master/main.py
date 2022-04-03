import os
import pickle
import time
import argparse

import torch

import tree_gan
from tree_gan.learning_utils import tree_gan_evaluate

torch.set_default_dtype(torch.float32)

def main(args):

    sql_data_dir = os.path.join('data', 'sql')
    sql_model_path = os.path.join('models', 'sql.model')
    sql_bnf_path = os.path.join(sql_data_dir, 'SQL_sample.bnf')
    sql_lark_path = os.path.join(sql_data_dir, 'sql_lang.lark')
    sql_text_dir = os.path.join(sql_data_dir, 'sqls')
    sql_action_getter_path = os.path.join(sql_data_dir, 'action_getter.pickle')
    sql_action_sequences_dir = os.path.join(sql_data_dir, 'actsqu')

    # If exists, load the last checkpoints from the model file path
    if args.pretrained:
        checkpoint = args.ckpt
        if checkpoint != "":
            if os.path.exists(checkpoint):
                sql_model_path = checkpoint
                with open(sql_model_path, 'rb') as f:
                    generator_ckp, discriminator_ckp = pickle.load(f)
            else:
                raise ValueError("can't find checkpoint file:" + checkpoint)
        elif os.path.exists(sql_model_path):
            with open(sql_model_path, 'rb') as f:
                generator_ckp, discriminator_ckp = pickle.load(f)
        else:
            generator_ckp, discriminator_ckp = None, None
    else:
        generator_ckp, discriminator_ckp = None, None

    ckpt_out = args.save_dir
    if ckpt_out != "":
        if os.path.exists(ckpt_out) and os.path.isdir(ckpt_out):
            model_save_path = os.path.join(ckpt_out, "sql.model")
        else:
            raise ValueError("there is no directory:" + ckpt_out)
    else:
        model_save_path = sql_model_path

    stats_path = args.stats
    if stats_path != "":
        if os.path.exists(stats_path) and os.path.isdir(stats_path):
            stats_save_path = stats_path
        else:
            raise ValueError("there is no directory:" + ckpt_out)
    else:
        stats_save_path = "stats"

    # ------------------HYPER PARAMETERS---------------------
    all_params = dict(
        # HYPER PARAMETERS WITH DEFAULT VALUES: (device, random_seed, generator_ckp, discriminator_ckp,
        # generator_kwargs, discriminator_kwargs)
        sql_lang_model_path=model_save_path,
        stats_save_path=stats_save_path,
        a_s_dataset=tree_gan.ActionSequenceDataset(sql_bnf_path, sql_lark_path, sql_text_dir,
                                                   sql_action_getter_path, sql_action_sequences_dir),
        generator_ckp=generator_ckp,
        discriminator_ckp=discriminator_ckp,
        generator_kwargs={'action_embedding_size': 128},
        discriminator_kwargs={'action_embedding_size': 128},
        num_data_loader_workers=1,
        max_total_step=200000,  # min number of steps to take during generator training
        initial_episode_timesteps=150,  # initial max time steps in one episode
        final_episode_timesteps=300,  # final max time steps in one episode (MUST NOT EXCEED 'buffer_timestep')
        episode_timesteps_log_order=0,
        gamma=0.99,  # discount factor
        gae_lambda=0.95,  # lambda value for td(lambda) returns
        eps_clip=0.2,  # clip parameter for PPO
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

        random_seed=1234,
        lr=1e-4,
        buffer_timestep=10000,
        lr_decay_order=5,
        k_epochs=5,
        buffer_to_batch_ratio=2,
        optimizer_betas=(0.5, 0.75),
        # PRE-TRAINING HYPER PARAMETERS
        pre_train_epochs=6,
        pre_train_batch_size=64,
        # DISCRIMINATOR TRAINING HYPER PARAMETERS
        discriminator_train_epochs=1,
        discriminator_train_batch_size=64,
        # GAN TRAINING HYPER PARAMETERS
        gan_epochs=10
    )
    # -------------------------------------------------------
    if args.small:
        all_params['max_total_step'] = 10000
        all_params['buffer_timestep'] = 5000
        all_params['pre_train_epochs'] = 1
        all_params['pre_train_batch_size'] = 8
        all_params['discriminator_train_batch_size'] = 8
        all_params['gan_epochs'] = 2

    mean_reward, (tree_gen, tree_dis), episode_reward_lists = tree_gan_evaluate(**all_params)

    # Save the current checkpoints to the model file path
    with open(model_save_path, 'wb') as f:
        pickle.dump((tree_gen.state_dict(), tree_dis.state_dict()), f)

    with torch.no_grad():
        # Generate an action sequence (equivalent to parse tree or text file)
        _, generated_actions, _, _, _ = tree_gen(max_sequence_length=all_params['final_episode_timesteps'])

    a_s_dataset = all_params['a_s_dataset']
    print('---------------------------------')
    print(a_s_dataset.action_getter.construct_text_partial(generated_actions.squeeze().cpu().numpy().tolist()))
    print('---------------------------------')


if __name__ == '__main__':
    START = time.process_time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--small', dest='small', action='store_true', help="train with small data")
    parser.set_defaults(small=False)
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help="train from pretrained weights")
    parser.set_defaults(pretrained=False)
    parser.add_argument("--ckpt", type=str, default="", help="train from pre-trained weights checkpoint")
    parser.add_argument("--save_dir", type=str, default="", help="output path for trained model weights")
    parser.add_argument("--stats", type=str, default="", help="output path for saving training stats(losses, rewards)")
    arguments = parser.parse_args()

    main(arguments)

    print('ELAPSED TIME (sec): ' + str(time.process_time() - START))
    print('---------------------------------')
