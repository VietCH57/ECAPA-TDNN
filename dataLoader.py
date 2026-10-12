'''
DataLoader for training

Changes w.r.t. the original (behaviour with default arguments is unchanged):
  * every augmentation constant is a constructor argument (exposed in config.py)
  * MUSAN / RIR folders are discovered automatically (musan_split or the raw MUSAN layout),
    and augmentations whose data is missing are disabled instead of crashing
  * noise files are read segment-wise (only the needed samples) -> fast with the raw, long MUSAN files
  * numpy.float (removed in NumPy >= 1.24) -> numpy.float64
'''

import glob, numpy, os, random, soundfile, torch
from scipy import signal

AUG_NAMES = ['clean', 'reverb', 'babble', 'music', 'noise', 'tv']


def read_list(train_list):
	'''Returns (speaker_ids, relative_paths) from a "<spk> <path>" list (space or tab separated).'''
	spks, paths = [], []
	for line in open(train_list).read().splitlines():
		if not line.strip():
			continue
		parts = line.split()
		spks.append(parts[0]); paths.append(parts[1])
	return spks, paths


def speaker_dict(train_list):
	'''Speaker -> integer label, identical to the original mapping (sorted speaker ids).'''
	spks, _ = read_list(train_list)
	keys = sorted(set(spks))
	return {key: ii for ii, key in enumerate(keys)}


def find_musan_root(path):
	'''Return the folder that contains music/ noise/ speech/ (searching up to 3 levels below path).'''
	if not path or not os.path.isdir(path):
		return None
	want = {'music', 'noise', 'speech'}
	for root, dirs, _ in os.walk(path):
		if want.issubset(set(dirs)):
			return root
		if root[len(path):].count(os.sep) >= 3:
			dirs[:] = []
	return None


def find_rir_root(path):
	'''Prefer a "simulated_rirs" folder below path (as in the original setup); otherwise use path itself.'''
	if not path or not os.path.isdir(path):
		return None
	if os.path.basename(os.path.normpath(path)) == 'simulated_rirs':
		return path
	for root, dirs, _ in os.walk(path):
		if 'simulated_rirs' in dirs:
			return os.path.join(root, 'simulated_rirs')
		if root[len(path):].count(os.sep) >= 3:
			dirs[:] = []
	return path


class train_loader(object):
	def __init__(self, train_list, train_path, musan_path, rir_path, num_frames,
				 aug_probs=(1, 1, 1, 1, 1, 1), snr_noise=(0, 15), snr_speech=(13, 20), snr_music=(5, 15),
				 num_noise=(1, 1), num_speech=(3, 8), num_music=(1, 1), verbose=True, **kwargs):
		self.train_path = train_path
		self.num_frames = num_frames
		self.length     = num_frames * 160 + 240
		# Load and configure augmentation files
		self.noisetypes = ['noise', 'speech', 'music']
		self.noisesnr = {'noise': list(snr_noise), 'speech': list(snr_speech), 'music': list(snr_music)}
		self.numnoise = {'noise': list(num_noise), 'speech': list(num_speech), 'music': list(num_music)}
		self.noiselist = {}
		musan_root = find_musan_root(musan_path)
		if musan_root is not None:
			for file in glob.glob(os.path.join(musan_root, '**', '*.wav'), recursive=True):
				cat = os.path.relpath(file, musan_root).split(os.sep)[0]
				if cat in self.noisetypes:
					self.noiselist.setdefault(cat, []).append(file)
		rir_root = find_rir_root(rir_path)
		self.rir_files = glob.glob(os.path.join(rir_root, '**', '*.wav'), recursive=True) if rir_root else []
		self._nframes = {}   # cache: noise file -> number of samples

		# Disable augmentations whose data is missing
		probs = [float(x) for x in aug_probs]
		need = {1: bool(self.rir_files), 2: 'speech' in self.noiselist, 3: 'music' in self.noiselist,
				4: 'noise' in self.noiselist, 5: 'speech' in self.noiselist and 'music' in self.noiselist}
		for k, ok in need.items():
			if not ok and probs[k] > 0:
				if verbose:
					print('[train_loader] "%s" augmentation disabled: data not found' % AUG_NAMES[k])
				probs[k] = 0.0
		if sum(probs) <= 0:
			probs[0] = 1.0
		self.aug_probs = probs
		if verbose:
			print('[train_loader] MUSAN root: %s | %s' % (musan_root, {k: len(v) for k, v in self.noiselist.items()}))
			print('[train_loader] RIR root: %s | %d files' % (rir_root, len(self.rir_files)))
			tot = sum(probs)
			print('[train_loader] augmentation probabilities: ' + ', '.join('%s=%.2f' % (n, p / tot) for n, p in zip(AUG_NAMES, probs)))

		# Load data & labels
		self.data_list  = []
		self.data_label = []
		dictkeys = speaker_dict(train_list)
		spks, paths = read_list(train_list)
		for spk, rel in zip(spks, paths):
			self.data_label.append(dictkeys[spk])
			self.data_list.append(os.path.join(train_path, rel))
		self.n_class = len(dictkeys)

	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		audio, sr = soundfile.read(self.data_list[index])
		length = self.length
		if audio.shape[0] <= length:
			shortage = length - audio.shape[0]
			audio = numpy.pad(audio, (0, shortage), 'wrap')
		start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
		audio = audio[start_frame:start_frame + length]
		audio = numpy.stack([audio],axis=0)
		# Data Augmentation
		augtype = random.choices(range(6), weights=self.aug_probs)[0]
		if augtype == 0:   # Original
			audio = audio
		elif augtype == 1: # Reverberation
			audio = self.add_rev(audio)
		elif augtype == 2: # Babble
			audio = self.add_noise(audio, 'speech')
		elif augtype == 3: # Music
			audio = self.add_noise(audio, 'music')
		elif augtype == 4: # Noise
			audio = self.add_noise(audio, 'noise')
		elif augtype == 5: # Television noise
			audio = self.add_noise(audio, 'speech')
			audio = self.add_noise(audio, 'music')
		return torch.FloatTensor(audio[0]), self.data_label[index]

	def __len__(self):
		return len(self.data_list)

	def add_rev(self, audio):
		rir_file    = random.choice(self.rir_files)
		rir, sr     = soundfile.read(rir_file)
		if rir.ndim > 1:
			rir = rir[:, 0]
		rir         = numpy.expand_dims(rir.astype(numpy.float64),0)
		rir         = rir / numpy.sqrt(numpy.sum(rir**2))
		return signal.convolve(audio, rir, mode='full')[:,:self.length]

	def _read_segment(self, file, length):
		'''Same result as reading the whole file, wrap-padding if short, then taking a random crop.'''
		n = self._nframes.get(file)
		if n is None:
			n = soundfile.info(file).frames
			self._nframes[file] = n
		if n <= length:
			noiseaudio, sr = soundfile.read(file)
			if noiseaudio.ndim > 1:
				noiseaudio = noiseaudio[:, 0]
			shortage = length - noiseaudio.shape[0]
			return numpy.pad(noiseaudio, (0, shortage), 'wrap')[:length]
		start_frame = numpy.int64(random.random()*(n-length))
		noiseaudio, sr = soundfile.read(file, start=int(start_frame), stop=int(start_frame) + length)
		if noiseaudio.ndim > 1:
			noiseaudio = noiseaudio[:, 0]
		return noiseaudio

	def add_noise(self, audio, noisecat):
		clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4)
		numnoise    = self.numnoise[noisecat]
		pool        = self.noiselist[noisecat]
		noiselist   = random.sample(pool, min(len(pool), random.randint(numnoise[0],numnoise[1])))
		noises = []
		for noise in noiselist:
			noiseaudio = self._read_segment(noise, self.length)
			noiseaudio = numpy.stack([noiseaudio],axis=0)
			noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4)
			noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio
