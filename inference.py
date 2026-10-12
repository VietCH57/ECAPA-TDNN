'''
Shared inference utilities: trial-list parsing, embedding extraction and scoring.

Embedding / scoring follow the original ECAPAModel.eval_network exactly:
  * embedding_1: whole utterance
  * embedding_2: `num_eval` crops of `eval_frames` frames, evenly spaced (wrap-padded if short)
  * both L2-normalised; score = (cos(full_a, full_b) + mean_ij cos(crop_a_i, crop_b_j)) / 2
The mean over all crop pairs equals the dot product of the mean crop embeddings, which is how
it is vectorised here (numerically identical).
'''

import os, numpy, soundfile, torch, tqdm
import torch.nn.functional as F

LABEL_WORDS = {'target': 1, 'nontarget': 0, 'non-target': 0, 'tar': 1, 'non': 0, 'true': 1, 'false': 0}


def parse_trials(path):
	'''Parse a trial list. Supported line formats (whitespace separated):
	     "1 a.wav b.wav"              (VoxCeleb, label first, 1 = same speaker)
	     "a.wav b.wav target"         (label last, target / nontarget or 1 / 0)
	     "a.wav b.wav"                (no label, e.g. a hidden test set)
	Returns (enroll, test, labels) where labels is a list of int or None.'''
	enroll, test, labels = [], [], []
	for n, line in enumerate(open(path).read().splitlines(), 1):
		parts = line.split()
		if not parts:
			continue
		if len(parts) == 2:
			a, b, lab = parts[0], parts[1], None
		elif len(parts) == 3 and parts[0] in ('0', '1'):
			lab, a, b = int(parts[0]), parts[1], parts[2]
		elif len(parts) == 3:
			a, b, w = parts
			if w.lower() in LABEL_WORDS:
				lab = LABEL_WORDS[w.lower()]
			elif w in ('0', '1'):
				lab = int(w)
			else:
				raise ValueError('%s:%d: cannot parse label %r' % (path, n, w))
		else:
			raise ValueError('%s:%d: expected 2 or 3 columns, got %r' % (path, n, line))
		enroll.append(a); test.append(b); labels.append(lab)
	if any(l is None for l in labels):
		labels = [None] * len(labels)
	return enroll, test, labels


def _crops(audio, eval_frames, num_eval):
	max_audio = eval_frames * 160 + 240
	if audio.shape[0] <= max_audio:
		shortage = max_audio - audio.shape[0]
		audio = numpy.pad(audio, (0, shortage), 'wrap')
	feats = []
	startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=num_eval)
	for asf in startframe:
		feats.append(audio[int(asf):int(asf)+max_audio])
	return numpy.stack(feats, axis=0).astype(numpy.float64)


@torch.no_grad()
def extract_embeddings(encoder, files, root, device='cuda', eval_frames=300, num_eval=5, mode='both', progress=True):
	'''Returns {file: (emb_full [D] or None, emb_crops_mean [D] or None)} as float32 numpy arrays.
	emb_crops_mean is the mean of the L2-normalised crop embeddings (NOT re-normalised).'''
	encoder.eval()
	out = {}
	it = tqdm.tqdm(files, total=len(files)) if progress else files
	for file in it:
		audio, _ = soundfile.read(os.path.join(root, file))
		if audio.ndim > 1:
			audio = audio.mean(1)
		e_full = e_crop = None
		if mode in ('both', 'full'):
			data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).to(device)
			e_full = F.normalize(encoder.forward(data_1, aug=False), p=2, dim=1)[0].float().cpu().numpy()
		if mode in ('both', 'crops'):
			data_2 = torch.FloatTensor(_crops(audio, eval_frames, num_eval)).to(device)
			e_crop = F.normalize(encoder.forward(data_2, aug=False), p=2, dim=1).mean(0).float().cpu().numpy()
		out[file] = (e_full, e_crop)
	return out


def score_trials(embeddings, enroll, test, mode='both'):
	'''Vectorised version of the original scoring loop.'''
	keys = sorted(set(enroll) | set(test))
	k2i = {k: i for i, k in enumerate(keys)}
	a = numpy.array([k2i[x] for x in enroll]); b = numpy.array([k2i[x] for x in test])
	scores = []
	if mode in ('both', 'full'):
		E = numpy.stack([embeddings[k][0] for k in keys])
		scores.append((E[a] * E[b]).sum(1))
	if mode in ('both', 'crops'):
		E = numpy.stack([embeddings[k][1] for k in keys])
		scores.append((E[a] * E[b]).sum(1))
	return numpy.mean(scores, axis=0)


def compute_metrics(scores, labels, p_targets=(0.05,)):
	'''EER (%) with the original tuneThresholdfromScore, minDCF with the original ComputeMinDcf.'''
	from tools import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf
	scores = list(map(float, scores)); labels = list(map(int, labels))
	EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
	fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
	res = {'EER': float(EER)}
	for p in p_targets:
		res['minDCF%g' % p] = float(ComputeMinDcf(fnrs, fprs, thresholds, p, 1, 1)[0])
	return res


def subset_metrics(scores, labels, enroll, test, meta_csv, p_target=0.05):
	'''EER on subsets defined by val_trials_meta.csv (from the split notebook):
	   O = targets + random non-targets, H = targets + same-gender non-targets,
	   X = cross-channel targets + same-gender same-channel non-targets, and O by min duration.'''
	import pandas as pd
	meta = pd.read_csv(meta_csv)
	key = pd.Series(range(len(meta)), index=pd.MultiIndex.from_arrays([meta.enroll, meta.test]))
	idx = key.reindex(pd.MultiIndex.from_arrays([enroll, test])).values
	if numpy.isnan(idx).any():
		print('[subset_metrics] %d trials not found in %s -> skipped' % (numpy.isnan(idx).sum(), meta_csv))
	ok = ~numpy.isnan(idx)
	m = meta.iloc[idx[ok].astype(int)].reset_index(drop=True)
	s = numpy.asarray(scores)[ok]; l = numpy.asarray(labels)[ok]
	t = (m.label == 1).values
	sub = {'O': t | (m.type == 'nontarget_rand').values,
		   'H': t | (m.type == 'nontarget_samegender').values,
		   'X': (t & m.cross_channel.values) | ((m.type == 'nontarget_samegender').values & ~m.cross_channel.values)}
	o = sub['O']
	for name, lo, hi in [('dur<2s', 0, 2), ('dur2-4s', 2, 4), ('dur4-8s', 4, 8), ('dur>=8s', 8, 1e9)]:
		sub['O_' + name] = o & (m.min_dur.values >= lo) & (m.min_dur.values < hi)
	res = {}
	for name, mask in sub.items():
		if mask.sum() and 0 < l[mask].sum() < mask.sum():
			r = compute_metrics(s[mask], l[mask], [p_target])
			res['EER_' + name] = r['EER']
			if name in ('O', 'H', 'X'):
				res['minDCF_' + name] = r['minDCF%g' % p_target]
	return res


def load_encoder_weights(encoder, path, device='cpu', verbose=True):
	'''Load encoder weights from a full ECAPAModel state_dict (model_XXXX.model / pretrain.model),
	   a training checkpoint (last.ckpt) or a bare encoder state_dict.'''
	state = torch.load(path, map_location=device)
	if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
		state = state['model']
	own = encoder.state_dict()
	loaded, skipped = 0, []
	for name, param in state.items():
		n = name.replace('module.', '')
		if n.startswith('speaker_encoder.'):
			n = n[len('speaker_encoder.'):]
		elif n.startswith('speaker_loss.'):
			continue
		if n in own and own[n].shape == param.shape:
			own[n].copy_(param); loaded += 1
		else:
			skipped.append(name)
	if verbose:
		print('Loaded %d/%d encoder tensors from %s%s' % (loaded, len(own), path, (' | skipped: %s' % skipped[:5]) if skipped else ''))
	if loaded != len(own):
		raise RuntimeError('Encoder weights incomplete (%d/%d). Check --C and the checkpoint.' % (loaded, len(own)))
	return encoder
