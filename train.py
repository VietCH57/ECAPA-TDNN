'''
Train ECAPA-TDNN with every hyper-parameter on the command line (or in a --config YAML/JSON).

    python train.py --train_list ... --train_path ... --eval_list ... --eval_path ... --save_path exps/exp1
    python train.py --config configs/vi_sred26.yaml --batch_size 200
    python train.py --eval --initial_model exps/exp1/model/best.model --eval_list ... --eval_path ...

Outputs in --save_path:
    config.json     all arguments of the run
    speakers.txt    speaker id -> class index
    score.txt       one line per evaluation (same format as the original + minDCF)
    model/model_XXXX.model   weights (same format as the original, loadable by trainECAPAModel.py)
    model/best.model         best EER so far
    model/last.ckpt          weights + optimizer + epoch (exact resume)
'''

import glob, os, random, re, time, warnings
import numpy, torch
from config import parse_args, save_config
from dataLoader import train_loader, speaker_dict
from ECAPAModel import ECAPAModel


def seed_everything(seed):
	random.seed(seed); numpy.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def worker_init(worker_id):
	# give every worker a different numpy seed (python's `random` is already seeded per worker by PyTorch)
	numpy.random.seed((torch.initial_seed() + worker_id) % 2**32)


def epoch_of(path):
	m = re.search(r'model_(\d+)\.model$', path)
	return int(m.group(1)) if m else -1


def main():
	warnings.simplefilter("ignore")
	torch.multiprocessing.set_sharing_strategy('file_system')
	args = parse_args()
	if args.device.startswith('cuda') and not torch.cuda.is_available():
		print('CUDA not available -> using CPU'); args.device = 'cpu'
	if args.seed >= 0:
		seed_everything(args.seed)

	args.score_save_path = os.path.join(args.save_path, 'score.txt')
	args.model_save_path = os.path.join(args.save_path, 'model')
	os.makedirs(args.model_save_path, exist_ok=True)

	## Number of classes from the training list
	if not args.eval:
		spk2id = speaker_dict(args.train_list)
		if args.n_class and args.n_class != len(spk2id):
			print('WARNING: --n_class %d != %d speakers in %s -> using %d' % (args.n_class, len(spk2id), args.train_list, len(spk2id)))
		args.n_class = len(spk2id)
		with open(os.path.join(args.save_path, 'speakers.txt'), 'w') as f:
			for k, v in spk2id.items():
				f.write('%s %d\n' % (k, v))
	elif not args.n_class:
		args.n_class = 5994   # only the encoder matters for evaluation; the head is skipped if its size differs

	save_config(args, os.path.join(args.save_path, 'config.json' if not args.eval else 'config_eval.json'))
	print('Arguments:'); [print('  %-14s %s' % (k, v)) for k, v in sorted(vars(args).items())]

	eval_kw = dict(eval_frames=args.eval_frames, num_eval=args.num_eval, eval_mode=args.eval_mode,
				   p_targets=args.dcf_ptarget, eval_meta=args.eval_meta, return_all=True)

	## Only do evaluation, the initial_model is necessary
	if args.eval:
		assert args.initial_model, '--eval needs --initial_model'
		s = ECAPAModel(**vars(args))
		print("Model %s loaded from previous state!" % args.initial_model)
		s.load_parameters(args.initial_model)
		res = s.eval_network(args.eval_list, args.eval_path, **eval_kw)
		print(' | '.join('%s %.4f' % (k, v) for k, v in res.items()))
		return

	## Data loader
	trainloader = train_loader(**vars(args))
	g = torch.Generator()
	if args.seed >= 0:
		g.manual_seed(args.seed)
	trainLoader = torch.utils.data.DataLoader(trainloader, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu,
											  drop_last=True, pin_memory=args.device.startswith('cuda'),
											  worker_init_fn=worker_init, generator=g, persistent_workers=args.n_cpu > 0)
	print('Training: %d utterances, %d speakers, %d batches/epoch' % (len(trainloader), args.n_class, len(trainLoader)))

	## Model, possibly resumed
	s = ECAPAModel(**vars(args))
	epoch, EERs, best = 1, [], float('inf')
	last_ckpt = os.path.join(args.model_save_path, 'last.ckpt')
	modelfiles = sorted(glob.glob('%s/model_0*.model' % args.model_save_path), key=epoch_of)
	resume = args.resume
	if resume not in ('auto', 'none'):
		if resume.endswith('.ckpt'):
			ck = s.load_checkpoint(resume); epoch = ck['epoch'] + 1; EERs = ck.get('EERs', []); best = ck.get('best', best)
		else:
			s.load_parameters(resume); epoch = max(epoch_of(resume), 0) + 1
		print('Resumed from %s, starting at epoch %d' % (resume, epoch))
	elif resume == 'auto' and os.path.exists(last_ckpt):
		ck = s.load_checkpoint(last_ckpt); epoch = ck['epoch'] + 1; EERs = ck.get('EERs', []); best = ck.get('best', best)
		print('Resumed from %s (model + optimizer), starting at epoch %d' % (last_ckpt, epoch))
	elif resume == 'auto' and modelfiles and not args.initial_model:
		print("Model %s loaded from previous state!" % modelfiles[-1])
		s.load_parameters(modelfiles[-1]); epoch = epoch_of(modelfiles[-1]) + 1
	elif args.initial_model:
		print("Model %s loaded from previous state!" % args.initial_model)
		s.load_parameters(args.initial_model)

	score_file = open(args.score_save_path, "a+")
	do_eval = bool(args.eval_list)

	t_start, ep_times = time.time(), []
	while epoch <= args.max_epoch:
		# stop cleanly before a wall-clock budget (e.g. Kaggle's 12 h) runs out; rerun the same command to resume
		if args.max_hours > 0 and ep_times:
			need = max(ep_times[-args.test_step:]) * 1.1
			if time.time() - t_start + need > args.max_hours * 3600:
				print('Stopping before epoch %d: --max_hours %.2f would be exceeded (%.2f h used). Rerun to resume from last.ckpt.'
					  % (epoch, args.max_hours, (time.time() - t_start) / 3600))
				break
		t0 = time.time()
		## Training for one epoch
		loss, lr, acc = s.train_network(epoch=epoch, loader=trainLoader)
		train_min = (time.time() - t0) / 60

		## Evaluation every [test_step] epochs
		if epoch % args.test_step == 0 or epoch == args.max_epoch:
			s.save_parameters(args.model_save_path + "/model_%04d.model" % epoch)
			msg = "%d epoch, LR %f, LOSS %f, ACC %2.2f%%" % (epoch, lr, loss, acc)
			if do_eval:
				res = s.eval_network(args.eval_list, args.eval_path, **eval_kw)
				EERs.append(res['EER'])
				if res['EER'] < best:
					best = res['EER']
					if args.save_best:
						s.save_parameters(args.model_save_path + "/best.model")
				dcf_key = 'minDCF%g' % args.dcf_ptarget[0]
				msg += ", EER %2.2f%%, bestEER %2.2f%%, %s %.4f" % (EERs[-1], min(EERs), dcf_key, res[dcf_key])
				extra = {k: v for k, v in res.items() if k not in ('EER', dcf_key)}
				if extra:
					msg += ", " + ", ".join("%s %.4f" % (k, v) for k, v in extra.items())
			print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, "| train %.1f min" % train_min)
			score_file.write(msg + "\n"); score_file.flush()
			if args.keep_last > 0:
				olds = sorted(glob.glob('%s/model_0*.model' % args.model_save_path), key=epoch_of)[:-args.keep_last]
				for f in olds:
					os.remove(f)
		s.save_checkpoint(last_ckpt, epoch, {'EERs': [float(x) for x in EERs], 'best': float(best)})
		ep_times.append(time.time() - t0)
		epoch += 1
	score_file.close()


if __name__ == '__main__':
	main()
