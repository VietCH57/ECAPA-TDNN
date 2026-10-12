'''
This part is used to train the speaker model and evaluate the performances

Changes w.r.t. the original (defaults reproduce the original behaviour):
  * device, weight decay, LR step, grad clipping and SpecAugment ranges are arguments
  * LR schedule: lr = lr0 * lr_decay ** ((epoch - 1) // lr_step)  (what StepLR.step(epoch - 1) did)
  * evaluation shares the vectorised code in inference.py; crop length / number / mode are arguments
  * full checkpoints (model + optimizer) for exact resuming
  * numpy.float (removed in NumPy >= 1.24) no longer used
The network (model.py) and the loss (loss.py) are untouched.
'''

import torch, sys, os, time
import torch.nn as nn
from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN
from inference import parse_trials, extract_embeddings, score_trials, compute_metrics, subset_metrics

class ECAPAModel(nn.Module):
	def __init__(self, lr, lr_decay, C , n_class, m, s, test_step, lr_step=None, weight_decay=2e-5,
				 device=None, specaug=True, specaug_freq=(0, 8), specaug_time=(0, 10), grad_clip=0.0,
				 log_interval=1, **kwargs):
		super(ECAPAModel, self).__init__()
		self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
		## ECAPA-TDNN
		self.speaker_encoder = ECAPA_TDNN(C = C).to(self.device)
		# SpecAugment settings are plain attributes of FbankAug -> set them without touching model.py
		self.speaker_encoder.specaug.freq_mask_width = tuple(specaug_freq)
		self.speaker_encoder.specaug.time_mask_width = tuple(specaug_time)
		self.use_specaug = specaug
		## Classifier
		self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).to(self.device)

		self.lr0, self.lr_decay = lr, lr_decay
		self.lr_step   = lr_step if lr_step else test_step
		self.grad_clip = grad_clip
		self.log_interval = max(1, log_interval)
		self.optim     = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = weight_decay)
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

	def lr_at(self, epoch):
		return self.lr0 * self.lr_decay ** ((epoch - 1) // self.lr_step)

	def train_network(self, epoch, loader):
		self.train()
		## Update the learning rate based on the current epoch
		lr = self.lr_at(epoch)
		for g in self.optim.param_groups:
			g['lr'] = lr
		index, top1, loss = 0, 0, 0
		n_batches = len(loader)
		for num, (data, labels) in enumerate(loader, start = 1):
			self.zero_grad()
			labels            = torch.LongTensor(labels).to(self.device)
			speaker_embedding = self.speaker_encoder.forward(data.to(self.device), aug = self.use_specaug)
			nloss, prec       = self.speaker_loss.forward(speaker_embedding, labels)
			nloss.backward()
			if self.grad_clip > 0:
				torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
			self.optim.step()
			index += len(labels)
			top1 += prec
			loss += nloss.detach().cpu().numpy()
			if num % self.log_interval == 0 or num == n_batches:
				sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
				" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / n_batches)) + \
				" Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
				sys.stderr.flush()
		sys.stdout.write("\n")
		return loss/num, lr, (top1/index*len(labels)).item() if torch.is_tensor(top1) else top1/index*len(labels)

	def eval_network(self, eval_list, eval_path, eval_frames=300, num_eval=5, eval_mode='both',
					 p_targets=(0.05,), eval_meta='', return_all=False):
		'''Original signature/return value kept: returns (EER, minDCF@p_targets[0]).
		With return_all=True returns a dict with all metrics (and subset metrics if eval_meta is given).'''
		self.eval()
		enroll, test, labels = parse_trials(eval_list)
		if labels[0] is None:
			raise ValueError('%s has no labels; use infer.py --mode score for unlabelled trials' % eval_list)
		files = sorted(set(enroll) | set(test))
		emb = extract_embeddings(self.speaker_encoder, files, eval_path, self.device, eval_frames, num_eval, eval_mode)
		scores = score_trials(emb, enroll, test, eval_mode)
		res = compute_metrics(scores, labels, p_targets)
		if eval_meta:
			res.update(subset_metrics(scores, labels, enroll, test, eval_meta, p_targets[0]))
		if return_all:
			return res
		return res['EER'], res['minDCF%g' % p_targets[0]]

	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	def save_checkpoint(self, path, epoch, extra=None):
		ckpt = {'model': self.state_dict(), 'optim': self.optim.state_dict(), 'epoch': epoch}
		if extra:
			ckpt.update(extra)
		tmp = path + '.tmp'
		torch.save(ckpt, tmp)
		os.replace(tmp, path)

	def load_checkpoint(self, path):
		ckpt = torch.load(path, map_location=self.device)
		self.load_state_dict(ckpt['model'])
		self.optim.load_state_dict(ckpt['optim'])
		return ckpt

	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path, map_location=self.device)
		if isinstance(loaded_state, dict) and 'model' in loaded_state and isinstance(loaded_state['model'], dict):
			loaded_state = loaded_state['model']
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("module.", "")
				if name not in self_state:
					print("%s is not in the model."%origname)
					continue
			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)
