import argparse
import time
import datetime

import numpy as np

import torch_ac
import tensorboardX
import sys

import utils
from utils import device
from model import ACModel, QModel


# Parse arguments

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo | dqn (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")

# hyper-parameters for dqn
parser.add_argument("--buffer_size", type=int, default=100000, help="size of replay buffer (default: 100000)")
parser.add_argument("--init_epsilon", type=float, default=1.0, help="initial epsilon for exploration (default: 1.0)")
parser.add_argument("--final_epsilon", type=float, default=0.1, help="final epsilon for exploration (default: 0.1)")
parser.add_argument("--zeta_epsilon", action='store_true', default=False, help="epsilon zeta for exploration (default: False)")
parser.add_argument("--simhash_repeat", action='store_true', default=False, help="use simhash for deciding action repeat (default: False)")
parser.add_argument("--init_expl", type=int, default=2000, help="initial exploration steps (default: 1000)")
parser.add_argument("--target_freq", type=int, default=100, help="update frequency for target network (default: 100)")
parser.add_argument("--eval_freq", type=int, default=2000, help="eval frequency (default: 2000)")
parser.add_argument("--load_model", default=None, help="name of the model to load")

if __name__ == "__main__":
    args = parser.parse_args()

    args.mem = args.recurrence > 1

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)
    if args.simhash_repeat:
        repeat_prob_list = []

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    txt_logger.info(f"Device: {device}\n")

    # Load environments

    if args.algo == 'dqn':
        args.procs = 1
        eval_envs = []
        for i in range(args.procs):
            eval_envs.append(utils.make_env(args.env, args.seed + 10000 * i))
    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.seed + 10000 * i))
    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor

    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model
    if args.algo == 'dqn':
        Q_network = QModel(obs_space, envs[0].action_space).to(device)
        target_network = QModel(obs_space, envs[0].action_space).to(device)
        txt_logger.info("{}\n".format(Q_network))
        Q_loaded_network = QModel(obs_space, envs[0].action_space).to(device)
        if args.load_model is not None:
            load_model_dir = utils.get_model_dir(args.load_model)
            load_model_status = utils.get_status(load_model_dir)
            Q_loaded_network.load_state_dict(load_model_status["model_state"])
            txt_logger.info("Model loaded\n")
            txt_logger.info("{}\n".format(Q_loaded_network))
    else:
        acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)
        if "model_state" in status:
            acmodel.load_state_dict(status["model_state"])
        acmodel.to(device)
        txt_logger.info("Model loaded\n")
        txt_logger.info("{}\n".format(acmodel))

    # Load algo

    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
    elif args.algo == 'dqn':
        algo = torch_ac.DQNAlgo(envs, eval_envs, Q_network, target_network, Q_loaded_network,
                                device, args.frames_per_proc, args.discount, args.lr,
                                args.optim_eps, args.epochs, args.batch_size, preprocess_obss, args.frames,
                                args.init_epsilon, args.final_epsilon, args.zeta_epsilon, args.simhash_repeat,
                                args.target_freq, args.buffer_size, args.init_expl)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < args.frames:
        # Update model parameters
        update_start_time = time.time()
        if args.algo == 'dqn':
            if num_frames % args.eval_freq == 0:
                reward = algo.eval()
                duration = int(time.time() - start_time)

                header = ["frames", "duration", 'return']
                data = [num_frames, duration, reward]

                txt_logger.info("F {:06} | D {} | R {:.2f}".format(*data))

                if num_frames == 0:
                    csv_logger.writerow(header)
                csv_logger.writerow(data)
                csv_file.flush()

                for field, value in zip(header, data):
                    tb_writer.add_scalar(field, value, num_frames)
            if args.simhash_repeat:
                num_frames, repeat_prob = algo.collect_experiences(num_frames)
                if repeat_prob is not None:
                    repeat_prob_list.append(repeat_prob)
                    if len(repeat_prob_list) % 1000 == 0:
                        np.savetxt(model_dir+'/repeat_prob.csv', np.array(repeat_prob_list), delimiter=",")
            else:
                num_frames = algo.collect_experiences(num_frames)

            if args.save_interval > 0 and num_frames % args.save_interval == 0:
                status = {"num_frames": num_frames, "update": update,
                          "model_state": Q_network.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
                if hasattr(preprocess_obss, "vocab"):
                    status["vocab"] = preprocess_obss.vocab.vocab
                utils.save_status(status, model_dir)
                txt_logger.info("Status saved")
        else:
            exps, logs1 = algo.collect_experiences()
            logs2 = algo.update_parameters(exps)
            logs = {**logs1, **logs2}
            update_end_time = time.time()

            num_frames += logs["num_frames"]
            update += 1

            # Print logs

            if update % args.log_interval == 0:
                fps = logs["num_frames"] / (update_end_time - update_start_time)
                duration = int(time.time() - start_time)
                return_per_episode = utils.synthesize(logs["return_per_episode"])
                rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
                num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

                header = ["update", "frames", "FPS", "duration"]
                data = [update, num_frames, fps, duration]
                header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
                data += rreturn_per_episode.values()
                header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                data += num_frames_per_episode.values()
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                    .format(*data))

                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()

                if status["num_frames"] == 0:
                    csv_logger.writerow(header)
                csv_logger.writerow(data)
                csv_file.flush()

                for field, value in zip(header, data):
                    tb_writer.add_scalar(field, value, num_frames)

            # Save status

            if args.save_interval > 0 and update % args.save_interval == 0:
                status = {"num_frames": num_frames, "update": update,
                          "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
                if hasattr(preprocess_obss, "vocab"):
                    status["vocab"] = preprocess_obss.vocab.vocab
                utils.save_status(status, model_dir)
                txt_logger.info("Status saved")
