'''
Command-line / config-file arguments for train.py.

All defaults reproduce the original hard-coded behaviour of this repo
(TaoRuijie/ECAPA-TDNN), so running without extra flags trains exactly as before.

Priority: command line  >  --config file (YAML or JSON)  >  defaults below.
'''

import argparse, json, os


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('1', 'true', 'yes', 'y', 't'):
        return True
    if v.lower() in ('0', 'false', 'no', 'n', 'f'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected, got %r' % v)


def _floats(v):
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return [float(x) for x in str(v).replace(' ', '').split(',') if x != '']


def _ints(v):
    return [int(round(x)) for x in _floats(v)]


def build_parser():
    p = argparse.ArgumentParser(description='ECAPA-TDNN trainer',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', type=str, default='', help='YAML/JSON file with any of the options below (command line overrides it)')

    g = p.add_argument_group('Data')
    g.add_argument('--train_list', type=str, default='/data08/VoxCeleb2/train_list.txt', help='Training list, each line: "<speaker> <relative/path.wav>" (space or tab)')
    g.add_argument('--train_path', type=str, default='/data08/VoxCeleb2/train/wav', help='Root folder of the training wavs')
    g.add_argument('--num_frames', type=int, default=200, help='Training crop length in frames (100 frames = 1 s)')
    g.add_argument('--batch_size', type=int, default=400, help='Batch size')
    g.add_argument('--n_cpu', type=int, default=4, help='Number of DataLoader workers')
    g.add_argument('--seed', type=int, default=-1, help='Random seed (-1 = do not seed, as in the original code)')

    g = p.add_argument_group('Augmentation (waveform)')
    g.add_argument('--musan_path', type=str, default='/data08/Others/musan_split', help='MUSAN root (the folder that contains music/ noise/ speech/, or any parent of it). Empty string = disable MUSAN')
    g.add_argument('--rir_path', type=str, default='/data08/Others/RIRS_NOISES/simulated_rirs', help='RIR root. If a "simulated_rirs" folder exists below it, only that folder is used. Empty string = disable RIR')
    g.add_argument('--aug_probs', type=_floats, default=[1, 1, 1, 1, 1, 1], help='Relative weights of [clean, reverb, babble, music, noise, tv(babble+music)]. Original: uniform')
    g.add_argument('--snr_noise', type=_floats, default=[0, 15], help='SNR range (dB) for MUSAN noise')
    g.add_argument('--snr_speech', type=_floats, default=[13, 20], help='SNR range (dB) for MUSAN speech (babble)')
    g.add_argument('--snr_music', type=_floats, default=[5, 15], help='SNR range (dB) for MUSAN music')
    g.add_argument('--num_noise', type=_ints, default=[1, 1], help='Number of noise files mixed (min,max)')
    g.add_argument('--num_speech', type=_ints, default=[3, 8], help='Number of babble speakers mixed (min,max)')
    g.add_argument('--num_music', type=_ints, default=[1, 1], help='Number of music files mixed (min,max)')

    g = p.add_argument_group('Augmentation (SpecAugment, inside the model; architecture unchanged)')
    g.add_argument('--specaug', type=_str2bool, default=True, help='Enable SpecAugment during training')
    g.add_argument('--specaug_freq', type=_ints, default=[0, 8], help='Frequency mask width range [min,max)')
    g.add_argument('--specaug_time', type=_ints, default=[0, 10], help='Time mask width range [min,max)')

    g = p.add_argument_group('Optimisation')
    g.add_argument('--max_epoch', type=int, default=80, help='Maximum number of epochs')
    g.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    g.add_argument('--lr_decay', type=float, default=0.97, help='Multiply LR by this every --lr_step epochs')
    g.add_argument('--lr_step', type=int, default=0, help='Decay LR every N epochs (0 = same as --test_step, i.e. original behaviour)')
    g.add_argument('--weight_decay', type=float, default=2e-5, help='Adam weight decay')
    g.add_argument('--grad_clip', type=float, default=0.0, help='Clip grad-norm to this value (0 = off, original)')

    g = p.add_argument_group('Model & loss')
    g.add_argument('--C', type=int, default=1024, help='Channel size of the ECAPA-TDNN encoder')
    g.add_argument('--m', type=float, default=0.2, help='AAM-softmax margin')
    g.add_argument('--s', type=float, default=30, help='AAM-softmax scale')
    g.add_argument('--n_class', type=int, default=0, help='Number of training speakers (0 = infer from --train_list)')

    g = p.add_argument_group('Evaluation during training')
    g.add_argument('--eval_list', type=str, default='/data08/VoxCeleb1/veri_test2.txt', help='Trial list: "label f1 f2" (label 1/0) or "f1 f2 target|nontarget". Empty = no evaluation')
    g.add_argument('--eval_path', type=str, default='/data08/VoxCeleb1/test/wav', help='Root folder of the evaluation wavs')
    g.add_argument('--eval_meta', type=str, default='', help='Optional val_trials_meta.csv -> also report EER on hard / cross-channel / duration subsets')
    g.add_argument('--test_step', type=int, default=1, help='Evaluate and save every N epochs')
    g.add_argument('--eval_frames', type=int, default=300, help='Crop length (frames) of the segmented-utterance embedding')
    g.add_argument('--num_eval', type=int, default=5, help='Number of crops per utterance')
    g.add_argument('--eval_mode', type=str, default='both', choices=['both', 'full', 'crops'], help='Score = mean(full, crops) [original] / full utterance only / crops only')
    g.add_argument('--dcf_ptarget', type=_floats, default=[0.05], help='P_target values for minDCF (first one is used for "best" tracking)')

    g = p.add_argument_group('Checkpoints & logging')
    g.add_argument('--save_path', type=str, default='exps/exp1', help='Output folder (score.txt, model/, config.json)')
    g.add_argument('--initial_model', type=str, default='', help='Start from these weights (epoch restarts at 1). Mismatched layers, e.g. the classifier head, are skipped')
    g.add_argument('--resume', type=str, default='auto', help='"auto" = continue from save_path if a checkpoint exists, "none" = never, or a path to last.ckpt / model_XXXX.model')
    g.add_argument('--max_hours', type=float, default=0, help='Stop (after the current epoch) before this many hours of training are used; 0 = no limit. Rerun to resume')
    g.add_argument('--keep_last', type=int, default=0, help='Keep only the N most recent model_XXXX.model files (0 = keep all, original)')
    g.add_argument('--save_best', type=_str2bool, default=True, help='Also save model/best.model (lowest EER)')
    g.add_argument('--log_interval', type=int, default=1, help='Print training progress every N batches (1 = original)')
    g.add_argument('--device', type=str, default='cuda', help='cuda / cuda:0 / cpu')
    g.add_argument('--data_parallel', type=_str2bool, default=False, help='Split each training batch over all visible GPUs (torch.nn.DataParallel); evaluation stays on one GPU')

    g = p.add_argument_group('Command')
    g.add_argument('--eval', dest='eval', action='store_true', help='Only evaluate --initial_model on --eval_list and exit')
    return p


def load_config_file(path):
    with open(path) as f:
        if path.endswith(('.yaml', '.yml')):
            import yaml
            return yaml.safe_load(f) or {}
        return json.load(f)


def parse_args(argv=None):
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        cfg = load_config_file(pre.config)
        known = {a.dest for a in parser._actions}
        unknown = [k for k in cfg if k not in known]
        if unknown:
            raise ValueError('Unknown keys in %s: %s' % (pre.config, unknown))
        # values from the file become the new defaults; they still go through the type converters
        for a in parser._actions:
            if a.dest in cfg and a.type is not None and cfg[a.dest] is not None:
                cfg[a.dest] = a.type(cfg[a.dest])
        parser.set_defaults(**cfg)
    args = parser.parse_args(argv)
    if args.lr_step <= 0:
        args.lr_step = args.test_step
    if len(args.aug_probs) != 6:
        parser.error('--aug_probs needs 6 values: clean,reverb,babble,music,noise,tv')
    return args


def save_config(args, path):
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
