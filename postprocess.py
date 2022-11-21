import os
import argparse
from itertools import permutations
import numpy as np
from mir_eval.separation import bss_eval_sources
from src.data import DatasetGenerator
import mindspore
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore import context, Tensor

parser = argparse.ArgumentParser('Evaluate separation performance using DPTNet')
parser.add_argument('--test_dir', type=str, default="/home/heu_MEDAI/623_project/src/ttjson/tt",
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--bin_path', type=str, default="/home/heu_MEDAI/623_project/src/result_Files/",
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--cal_sdr', type=int, default=0,
                    help='Whether calculate SDR, add this option because calculation of SDR is very slow')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--batch_size', default=3, type=int,
                    help='Batch size')
parser.add_argument('--segment', default=4, type=int,
                    help='The hidden size of RNN')

EPS = 1e-8

def evaluate(args, list1):
    total_SISNRi = 0
    total_SDRi = 0
    total_cnt = 0
    # Load data
    tt_dataset = DatasetGenerator(args.test_dir, args.batch_size,
                                  sample_rate=args.sample_rate, segment=args.segment)
    tt_loader = ds.GeneratorDataset(tt_dataset, ["mixture", "lens", "sources"], shuffle=False)
    tt_loader = tt_loader.batch(batch_size=1)

    i = 0
    for data in tt_loader.create_dict_iterator():
        padded_mixture = data["mixture"]
        mixture_lengths = data["lens"]
        padded_source = data["sources"]
        padded_mixture = ops.Cast()(padded_mixture, mindspore.float32)
        padded_source = ops.Cast()(padded_source, mindspore.float32)
        mixture_lengths_with_list = mixture_lengths.asnumpy().tolist()
        estimate_source = list1[i]
        i += 1

        _, estimate_source, reorder_estimate_source = \
            cal_loss(padded_source, estimate_source, mixture_lengths)
        mixture = remove_pad(padded_mixture, mixture_lengths_with_list)
        source = remove_pad(padded_source, mixture_lengths_with_list)
        # NOTE: use reorder estimate source
        estimate_source = remove_pad(reorder_estimate_source,
                                     mixture_lengths_with_list)
        # for each utterance
        for mix, src_ref, src_est in zip(mixture, source, estimate_source):
            print("Utt", total_cnt + 1)
            # Compute SDRi
            if args.cal_sdr:
                avg_SDRi = cal_SDRi(src_ref, src_est, mix)
                total_SDRi += avg_SDRi
                print("\tSDRi={0:.2f}".format(avg_SDRi))
            # Compute SI-SNRi
            avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
            print("\tSI-SNRi={0:.2f}".format(avg_SISNRi))
            total_SISNRi += avg_SISNRi
            total_cnt += 1
    if args.cal_sdr:
        print("Average SDR improvement: {0:.2f}".format(total_SDRi / total_cnt))
    print("Average SISNR improvement: {0:.2f}".format(total_SISNRi / total_cnt))


def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    src_anchor = np.stack([mix, mix], axis=0)
    sdr, _, _, _ = bss_eval_sources(src_ref, src_est)
    sdr0, _, _, _ = bss_eval_sources(src_ref, src_anchor)
    avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
    # print("SDRi1: {0:.2f}, SDRi2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[1]))
    return avg_SDRi


def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    sisnr1 = cal_SISNR(src_ref[0], src_est[0])
    sisnr2 = cal_SISNR(src_ref[1], src_est[1])
    sisnr1b = cal_SISNR(src_ref[0], mix)
    sisnr2b = cal_SISNR(src_ref[1], mix)
    avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    return avg_SISNRi


def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calculate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr


def remove_pad(inputs, inputs_lengths):
    """
    Args:
        inputs: torch.Tensor, [B, C, T] or [B, T], B is batch size
        inputs_lengths: torch.Tensor, [B]
    Returns:
        results: a list containing B items, each item is [C, T], T varies
    """
    results = []
    dim = inputs.ndim
    if dim == 3:
        C = inputs.shape[1]
    for i, input_data in enumerate(inputs):
        if dim == 3:    # [B, C, T]
            results.append(input_data[:, :inputs_lengths[i]].view(C, -1).asnumpy())
        elif dim == 2:  # [B, T]
            results.append(input_data[:inputs_lengths[i]].view(-1).asnumpy())
    return results

def cal_loss(source, estimate_source, source_lengths):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    max_snr, perms, max_snr_idx = cal_si_snr_with_pit(source, estimate_source, source_lengths)
    mean = ops.ReduceMean()
    loss = 0 - mean(max_snr)
    reorder_estimate_source = reorder_source(estimate_source, perms, max_snr_idx)
    return loss, estimate_source, reorder_estimate_source


def cal_si_snr_with_pit(source, estimate_source_0, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    B, C, _ = source.shape
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source_1 = estimate_source_0 * mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).astype(mindspore.float32)
    sum_ = ops.ReduceSum(keep_dims=True)
    mean_target = sum_(source, 2) / num_samples
    mean_estimate = sum_(estimate_source_1, 2) / num_samples
    zero_mean_target_0 = source - mean_target
    zero_mean_estimate_0 = estimate_source_1 - mean_estimate
    # mask padding position along T
    zero_mean_target = zero_mean_target_0 * mask
    zero_mean_estimate = zero_mean_estimate_0 * mask
    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    expand_dims_0 = ops.ExpandDims()
    s_target = expand_dims_0(zero_mean_target, 1)  # [B, 1, C, T]
    s_estimate = expand_dims_0(zero_mean_estimate, 2)  # [B, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot_0 = sum_(s_estimate * s_target, 3)  # [B, C, C, 1]
    s_target_energy_0 = sum_(s_target * s_target, 3) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot_0 * s_target / s_target_energy_0  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    _sum = ops.ReduceSum(keep_dims=False)
    pair_wise_si_snr_0 = _sum(pair_wise_proj * pair_wise_proj, 3) / (_sum(e_noise * e_noise, 3) + EPS)
    log = ops.Log()
    log_10 = Tensor(np.array([10.0]), mindspore.float32)
    temp = log(log_10) / 10
    pair_wise_si_snr_1 = log(pair_wise_si_snr_0 + EPS)
    pair_wise_si_snr = pair_wise_si_snr_1 / temp # [B, C, C]

    # Get max_snr of each utterance
    # permutations, [C!, C]
    perms = Tensor(list(permutations(range(2))), dtype=mindspore.int64)
    perms_one_hot = perms_one_hot = Tensor(np.array([[1, 0], [0, 1], [0, 1], [1, 0]]), mindspore.float32)
    matmul = ops.MatMul()
    snr_set = matmul(pair_wise_si_snr.view(B, -1), perms_one_hot)
    Argmax = ops.Argmax(axis=1, output_type=mindspore.int32)
    max_snr_idx = Argmax(snr_set)  # [B]
    argmax = ops.ArgMaxWithValue(axis=1, keep_dims=True)
    _, max_snr = argmax(snr_set)
    max_snr /= C
    return max_snr, perms, max_snr_idx

def reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, C, T]
        perms: [C!, C], permutations
        max_snr_idx: [B], each item is between [0, C!)
    Returns:
        reorder_source: [B, C, T]
    """
    B, C, _ = source.shape
    # [B, C], permutation whose SI-SNR is max of each utterance
    # for each utterance, reorder estimate source according this permutation
    max_snr_perm = perms[max_snr_idx, :]
    zeros_like = ops.ZerosLike()
    reorder_source1 = zeros_like(source)
    for b in range(B):
        for c in range(C):
            if max_snr_perm[b][c] == 1:
                reorder_source1[b, c] = source[b, 1]
            else:
                reorder_source1[b, c] = source[b, 0]
    return reorder_source1


def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, _, T = source.shape
    ones = ops.Ones()
    mask = ones((B, 1, T), mindspore.float32)
    for i in range(B):
        mask[i, :, 32000:] = 0
    return mask

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    arg = parser.parse_args()
    audio_files = os.listdir(arg.bin_path)
    audio_files = sorted(audio_files, key=lambda x: int(os.path.splitext(x)[0]))

    list_ = []
    for f in audio_files:
        f_name = os.path.join(arg.bin_path, f.split('.')[0] + '.bin')
        logits = np.fromfile(f_name, np.float32).reshape(1, 2, 32000)
        logits = Tensor(logits)
        list_.append(logits)
    evaluate(arg, list_)
