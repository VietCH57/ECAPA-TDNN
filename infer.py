'''
Inference with a trained ECAPA-TDNN.

Modes
  score : trial list (labelled or not) -> "file1 file2 score" per line, SAME order as the trial list.
          If the list has labels (e.g. public test "a.wav b.wav target"), EER/minDCF are printed too.
          --zip writes results.zip containing results.txt (challenge submission format).
  eval  : like score, but labels are required and only metrics are printed (+ --eval_meta subsets).
  embed : embeddings for every file in --file_list (or every file of --trials) -> .npz

Examples
  python infer.py --mode score --model exps/exp1/model/best.model \
      --trials /kaggle/input/.../public-test.txt --data_path /kaggle/input/.../data \
      --out /kaggle/working/results.txt --zip
  python infer.py --mode eval --model exps/exp1/model/best.model --trials val_trials.txt --data_path .../data
  python infer.py --mode embed --model ... --file_list val_split.txt --data_path .../data --out emb.npz
'''

import argparse, json, os, time, zipfile
import numpy, torch
from model import ECAPA_TDNN
from inference import (parse_trials, extract_embeddings, score_trials, compute_metrics,
					   subset_metrics, load_encoder_weights)


def build_parser():
	p = argparse.ArgumentParser(description='ECAPA-TDNN inference', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	p.add_argument('--mode', choices=['score', 'eval', 'embed'], default='score')
	p.add_argument('--model', required=True, help='model_XXXX.model / best.model / last.ckpt / pretrain.model')
	p.add_argument('--C', type=int, default=1024, help='Channel size used when training the model')
	p.add_argument('--trials', type=str, default='', help='Trial list ("label a b", "a b target|nontarget" or "a b")')
	p.add_argument('--file_list', type=str, default='', help='[embed] list of files (last column of each line is used)')
	p.add_argument('--data_path', type=str, required=True, help='Root folder the paths in the list are relative to')
	p.add_argument('--out', type=str, default='results.txt', help='[score] score file / [embed] .npz file')
	p.add_argument('--zip', action='store_true', help='[score] also write <out_dir>/results.zip with results.txt inside')
	p.add_argument('--score_fmt', type=str, default='%.6f', help='[score] number format of the score')
	p.add_argument('--eval_frames', type=int, default=300, help='Crop length in frames for the crop embedding')
	p.add_argument('--num_eval', type=int, default=5, help='Number of crops per utterance')
	p.add_argument('--eval_mode', type=str, default='both', choices=['both', 'full', 'crops'])
	p.add_argument('--dcf_ptarget', type=str, default='0.05,0.01', help='P_target values for minDCF')
	p.add_argument('--eval_meta', type=str, default='', help='val_trials_meta.csv for subset EERs')
	p.add_argument('--emb_cache', type=str, default='', help='.npz to load embeddings from if it exists, otherwise to save them to')
	p.add_argument('--device', type=str, default='cuda')
	return p


def save_emb(path, emb):
	keys = sorted(emb)
	full = numpy.stack([emb[k][0] for k in keys]) if emb[keys[0]][0] is not None else numpy.zeros((0,))
	crop = numpy.stack([emb[k][1] for k in keys]) if emb[keys[0]][1] is not None else numpy.zeros((0,))
	numpy.savez(path, files=numpy.array(keys), full=full, crops=crop)


def load_emb(path):
	z = numpy.load(path, allow_pickle=False)
	files = list(z['files'])
	full = z['full'] if z['full'].size else None
	crop = z['crops'] if z['crops'].size else None
	return {f: (full[i] if full is not None else None, crop[i] if crop is not None else None) for i, f in enumerate(files)}


def main():
	args = build_parser().parse_args()
	if args.device.startswith('cuda') and not torch.cuda.is_available():
		print('CUDA not available -> using CPU'); args.device = 'cpu'
	p_targets = [float(x) for x in args.dcf_ptarget.split(',') if x]
	for path in (args.out, args.emb_cache):
		if path:
			os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

	## Files to embed
	if args.mode in ('score', 'eval'):
		assert args.trials, '--trials is required for mode %s' % args.mode
		enroll, test, labels = parse_trials(args.trials)
		files = sorted(set(enroll) | set(test))
		print('%d trials, %d unique files, labels: %s' % (len(enroll), len(files), labels[0] is not None))
		if args.mode == 'eval' and labels[0] is None:
			raise SystemExit('mode eval needs a labelled trial list')
	else:
		src = args.file_list or args.trials
		assert src, '--file_list (or --trials) is required for mode embed'
		if args.file_list:
			files = sorted({l.split()[-1] for l in open(args.file_list).read().splitlines() if l.strip()})
		else:
			e, t, _ = parse_trials(args.trials); files = sorted(set(e) | set(t))
	missing = [f for f in files if not os.path.isfile(os.path.join(args.data_path, f))]
	if missing:
		raise SystemExit('%d files not found under %s, e.g. %s' % (len(missing), args.data_path, missing[:3]))

	## Embeddings (from cache or the model)
	emb = None
	if args.emb_cache and os.path.exists(args.emb_cache):
		emb = load_emb(args.emb_cache)
		if not set(files) <= set(emb):
			print('Cache %s does not cover all files -> recomputing' % args.emb_cache); emb = None
		else:
			print('Embeddings loaded from', args.emb_cache)
	if emb is None:
		encoder = ECAPA_TDNN(C=args.C)
		load_encoder_weights(encoder, args.model, 'cpu')
		encoder = encoder.to(args.device)
		t0 = time.time()
		emb = extract_embeddings(encoder, files, args.data_path, args.device, args.eval_frames, args.num_eval, args.eval_mode)
		print('Embedded %d files in %.1f min' % (len(files), (time.time() - t0) / 60))
		if args.emb_cache:
			save_emb(args.emb_cache, emb)

	if args.mode == 'embed':
		save_emb(args.out, emb); print('Saved', args.out); return

	## Scores
	scores = score_trials(emb, enroll, test, args.eval_mode)
	if args.mode == 'score':
		with open(args.out, 'w') as f:
			for a, b, sc in zip(enroll, test, scores):
				f.write('%s %s %s\n' % (a, b, args.score_fmt % sc))
		# sanity check: same pairs, same order, one line per trial
		lines = open(args.out).read().splitlines()
		assert len(lines) == len(enroll) and all(l.split()[:2] == [a, b] for l, a, b in zip(lines, enroll, test))
		print('Wrote %d lines to %s' % (len(lines), args.out))
		if args.zip:
			zpath = os.path.join(os.path.dirname(os.path.abspath(args.out)), 'results.zip')
			with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
				z.write(args.out, arcname='results.txt')
			print('Wrote', zpath, '(contains results.txt)')
	if labels[0] is not None:
		res = compute_metrics(scores, labels, p_targets)
		if args.eval_meta:
			res.update(subset_metrics(scores, labels, enroll, test, args.eval_meta, p_targets[0]))
		print(' | '.join('%s %.4f' % (k, v) for k, v in res.items()))
		if args.mode == 'eval' or args.out:
			mpath = os.path.splitext(os.path.abspath(args.out))[0] + '_metrics.json'
			json.dump(res, open(mpath, 'w'), indent=2); print('Metrics saved to', mpath)


if __name__ == '__main__':
	main()
