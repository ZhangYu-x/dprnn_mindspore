import os
import argparse
from src.preprocess import preprocess
from src.data import DatasetGenerator
from src.network_define import WithLossCell
from src.Loss import loss
from src.model import Dual_RNN_model
from train_wrapper import TrainingWrapper
import mindspore.dataset as ds
from mindspore import load_checkpoint, load_param_into_net
from mindspore import nn, context, Model
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode

parser = argparse.ArgumentParser(
        description='Parameters for training Dual-Path-RNN')

parser.add_argument('--in_dir', type=str, default=r"/home/work/user-job-dir/inputs/data/",
                    help='Directory path of wsj0 including tr, cv and tt')
parser.add_argument('--out_dir', type=str, default=r"/home/work/user-job-dir/inputs/data_json",
                    help='Directory path to put output files')
parser.add_argument('--data_url',
                    help='path to training/inference dataset folder',
                    default='/home/work/user-job-dir/inputs/data/')

parser.add_argument('--train_url',
                    help='model folder to save/load',
                    default='/home/work/user-job-dir/model/')
# Dataset path on openi
# parser.add_argument('--train_dir', type=str, default="/home/work/user-job-dir/inputs/data_json/tr",
#                     help='directory including mix.json, s1.json and s2.json')
# Dataset path on 910
parser.add_argument('--train_dir', type=str, default="/mass_data/dataset/LS-2mix/Libri2Mix/tr",
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir', type=str, default='/mass_data/dataset/LS-2mix/Libri2Mix/cv',
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--segment', default=4, type=float,
                    help='Segment length (seconds)')
parser.add_argument('--data_batch_size', type=int, default=3,
                    help='Determine the number of available data')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Batch size of training data')
# Network architecture
parser.add_argument('--in_channels', default=64, type=int,
                    help='The number of expected features in the input')
parser.add_argument('--out_channels', default=64, type=int,
                    help='The number of features in the hidden state')
parser.add_argument('--hidden_channels', default=128, type=int,
                    help='The hidden size of RNN')
parser.add_argument('--bn_channels', default=128, type=int,
                    help='Number of channels after the conv1d')
parser.add_argument('--kernel_size', default=2, type=int,
                    help='Encoder and Decoder Kernel size')
parser.add_argument('--rnn_type', default='LSTM', type=str,
                    help='RNN, LSTM, GRU')
parser.add_argument('--norm', default='gln', type=str,
                    help='gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"')
parser.add_argument('--dropout', default=0.0, type=float,
                    help='dropout')
parser.add_argument('--num_layers', default=6, type=int,
                    help='Number of Dual-Path-Block')
parser.add_argument('--K', default=250, type=int,
                    help='The length of chunk')
parser.add_argument('--num_spks', default=2, type=int,
                    help='The number of speakers')

# optimizer
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Init learning rate')
parser.add_argument('--lr1', default=5e-4, type=float,
                    help='Init learning rate')
parser.add_argument('--lr2', default=2.5e-4, type=float,
                    help='Init learning rate')
parser.add_argument('--lr3', default=1.25e-4, type=float,
                    help='Init learning rate')
parser.add_argument('--l2', default=1e-5, type=float,
                    help='weight decay (L2 penalty)')

# save and load model
parser.add_argument('--save_folder', default='/home/heu_MEDAI/commit910_new/ckpt',
                    help='Location to save epoch models')
parser.add_argument("--device_id", type=int, default=6,
                    help="device id, default: 0.")
parser.add_argument("--device_num", type=int, default=2,
                    help="number of device, default: 0.")
parser.add_argument('--device_target', type=str, default='Ascend', choices=('Ascend', 'GPU', 'CPU'),
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--run_distribute', type=int, default=1,
                    help='Distribute or not')
parser.add_argument('--modelArts', default=0, type=int,
                    help='Whether to conduct cloud training')
parser.add_argument('--continue_train', default=0, type=int,
                    help='Continue from checkpoint model')
parser.add_argument('--ckpt_path', type=str, default="DPRNN-15_2555-1e-3.ckpt",
                    help='Path to model file created by training')

def main(args):
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    device_num = int(os.environ.get("RANK_SIZE", 1))
    if device_num == 1:
        is_distributed = 'False'
    elif device_num > 1:
        is_distributed = 'True'

    if is_distributed == 'True':
        print("parallel init", flush=True)
        init()
        rank_id = get_rank()
        context.reset_auto_parallel_context()
        parallel_mode = ParallelMode.DATA_PARALLEL
        rank_size = get_group_size()
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=args.device_num)
        context.set_auto_parallel_context(parameter_broadcast=True)
        print("Starting traning on multiple devices...")
    else:
        if args.modelArts:
            init()
            rank_id = get_rank()
            rank_size = get_group_size()
        else:
            context.set_context(device_id=args.device_id)

    if args.modelArts:
        import moxing as mox
        obs_data_url = args.data_url
        args.data_url = '/home/work/user-job-dir/inputs/data/'
        obs_train_url = args.train_url

        home = os.path.dirname(os.path.realpath(__file__))
        #save model
        save_folder = os.path.join(home, 'checkpoints') + str(rank_id)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        save_checkpoint_path = save_folder + '/device_' + os.getenv('DEVICE_ID') + '/'
        if not os.path.exists(save_checkpoint_path):
            os.makedirs(save_checkpoint_path)
        save_ckpt = os.path.join(save_checkpoint_path, 'dprnnCkpt')

        mox.file.copy_parallel(obs_data_url, args.data_url)
        print("Successfully Download {} to {}".format(obs_data_url, args.data_url))

        print("start preprocess on modelArts....")
        preprocess(args)
    # build dataloader
    print("Start datasetgenerator")
    tr_dataset = DatasetGenerator(args.train_dir, args.data_batch_size,
                                  sample_rate=args.sample_rate, segment=args.segment)
    if is_distributed == 'True':
        tr_loader = ds.GeneratorDataset(tr_dataset, ["mixture", "lens", "sources"],
                                        shuffle=True, num_shards=rank_size, shard_id=rank_id)
    else:
        tr_loader = ds.GeneratorDataset(tr_dataset, ["mixture", "lens", "sources"],
                                        shuffle=True)
    tr_loader = tr_loader.batch(args.batch_size)
    num_steps = tr_loader.get_dataset_size()
    print("data loading done")
    # build model
    net = Dual_RNN_model(args.in_channels, args.out_channels, args.hidden_channels, args.bn_channels,
                         bidirectional=True, norm=args.norm, num_layers=args.num_layers, dropout=args.dropout, K=args.K)
    if args.continue_train:
        if args.modelArts:
            home = os.path.dirname(os.path.realpath(__file__))
            ckpt = os.path.join(home, args.ckpt_path)
            params = load_checkpoint(ckpt)
            load_param_into_net(net, params)
        else:
            params = load_checkpoint(args.ckpt_path)
            load_param_into_net(net, params)
    print(net)
    # build optimizer
    milestone = [15 * num_steps, 45 * num_steps, 75 * num_steps, 100 * num_steps]
    learning_rates = [args.lr, args.lr1, args.lr2, args.lr3]
    lr = nn.piecewise_constant_lr(milestone, learning_rates)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr, weight_decay=args.l2)
    my_loss = loss()
    net_with_loss = WithLossCell(net, my_loss)
    net_with_loss_ = TrainingWrapper(net_with_loss, optimizer)

    net_with_loss_.set_train()

    model = Model(net_with_loss_)

    time_cb = TimeMonitor()
    loss_cb = LossMonitor(1)
    cb = [time_cb, loss_cb]

    config_ck = CheckpointConfig(save_checkpoint_steps=num_steps, keep_checkpoint_max=5)
    if args.modelArts:
        ckpt_cb = ModelCheckpoint(prefix="DPRNN", directory=save_ckpt, config=config_ck)
    else:
        if is_distributed == 'True' and args.device_id == 0:
            ckpt_cb = ModelCheckpoint(prefix="DPRNN", directory=args.save_folder, config=config_ck)
            cb += [ckpt_cb]
        if is_distributed == 'False':
            ckpt_cb = ModelCheckpoint(prefix="DPRNN", directory=args.save_folder, config=config_ck)
            cb += [ckpt_cb]

    model.train(epoch=100, train_dataset=tr_loader, callbacks=cb, dataset_sink_mode=True)
    if args.modelArts:
        import moxing as mox
        mox.file.copy_parallel(save_folder, obs_train_url)
        print("Successfully Upload {} to {}".format(save_folder, obs_train_url))

if __name__ == '__main__':
    arg = parser.parse_args()
    print(arg)
    main(arg)
