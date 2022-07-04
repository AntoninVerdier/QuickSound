import os
import argparse
import numpy as np
import scipy.signal as cignal
from scipy.io import wavfile
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

class Sound():
	""" Class for creating sound paradigm with multiple features

	Attributes
	----------
	name : str
		Sample's name. Will be used for saving
	signal : array
		Sound signal
	freq : dict
		Dictionnary containing frequencies information for quick access. Not yet supported with __add__ method.
	samplerate : int
		Number of samples needed in the isgnal per unit of time

	Constructors
	------------
	__init__(self, samplerate=192000)
		Initialize object and attribute
	__add__(self, other)
		Allow user to concatenate two Sound's signals
	__mul__(self, other)
		Allow user to overlap two Sound's signals

	Methods
	-------
	delay(self, duration)
		Generate a silence for a given duration
	noise(self, duration)
		Generate a white noise for a given duration
	colored_noise(self, duration, type='pink')
		Generate a colored noise for a given duration
	pure_tone(self, frequency, duration=1000)
		Generate a pure tone signal for a given duration
	freq_modulation(self, start_freq, end_freq, duration=2500)
		 a signal of increase or decreasing frequencies
	amplitude_modulation(self, freq, am_freq, duration=5000)
		Generate an aplitude-modulated tone at a reference frequency
	freq_noise(self, freq, noise_vol, duration=2500)
		Create a pure tone with noise in the background of increasing intensity
	multi_freqs(self, freqs, duration=2500)
		Generate multiple frequency harmonics
	harmonics(self, base_freq, patterns, duration=500)
		Generate patterns of harmonics
	steps(self, start_freq, end_freq, nstep, spacing='Log', duration=500)
		Generate a frequency modulated tone in steps
	save_wav(self, name=None):
		Save the signal as a .wav file

	"""
	def __init__(self, samplerate=192000, amplitude=70):

		""" Constructor at initialization
		"""
		self.name = 'test'
		self.signal = None
		self.freq = None
		self.samplerate = samplerate

		dBref = 100
		A=10**((amplitude-dBref)/20)
		
		self.amplitude = A

	def __add__(self, other):
		"""Define how to assemble generated sounds
		"""
		assert self.samplerate == other.samplerate, 'Signals must have the same samplerate'
		assert (self.signal is not None) & (other.signal is not None), 'Signals must be defined'

		newSound = Sound(samplerate=self.samplerate)
		newSound.signal = np.concatenate((self.signal, other.signal))

		return newSound

	def __mul__(self, other):
		""" Define how to combine two sounds
		"""
		assert self.samplerate == other.samplerate, 'Signales must have the same samplerate'
		assert (self.signal is not None) & (other.signal is not None), 'Signals must be defined'

		newSound = Sound(samplerate=self.samplerate)

		if len(self.signal) >= len(other.signal):
			self.signal = self.signal[:len(other.signal)]
		else:
			other.signal = other.signal[:len(self.signal)]

		newSound.signal = self.signal + other.signal

		return newSound


	def delay(self, duration):
		""" Generate a silence for a given duration

		Parameters
		----------
		duration : int
			Duration of the delay in ms
		"""
		sample = int(duration * 0.001 * self.samplerate)
		self.signal = np.array(np.zeros(sample))

	def noise(self, duration):
		""" generate a white noise for a given duration

		Parameters
		----------
		duration : int
			Duration of the noise in ms
		"""
		sample = int(duration * 0.001 * self.samplerate)
		self.signal = self.amplitude * np.random.normal(0, 1, size=sample)

	def colored_noise(self, duration, type='pink', freqs=[500, 1500]):
		""" generate a white noise for a given duration

		Parameters
		----------
		duration : int
			Duration of the noise in ms
		type : str, optional
			Type of colored noise. Only pink is available
		"""

		# WIP
		sample = int(duration * 0.001 * self.samplerate)
		sig = np.random.normal(0, 1, size=sample)
		if type=='pink':
			sos = cignal.butter(1, freqs, 'bandpass', fs=self.samplerate, output='sos')
		
		filtered = cignal.sosfilt(sos, sig)

		self.signal = self.amplitude * filtered


	def pure_tone(self, frequency,  duration=500):

		"""Generate a pure tone signal for a given duration

		Parameters
		----------
		frequency : int
			Frequency of the pure tone to generate
		duration : int, optional
			Duration of the tone in ms
		"""
		sample = int(duration * 0.001 * self.samplerate)
		time = np.arange(sample)
		pure_tone = self.amplitude * np.sin(2 * np.pi * frequency * time / self.samplerate)

		self.signal = np.array(pure_tone)
		self.freq = {'simple' : frequency}

	def freq_modulation(self, start_freq, end_freq, duration=500):
		"""Generate a signal of increase or decreasing frequencies

		Parameters
		----------
		start_freq
			Starting frequency of the signal
		end_freq
			Ending frequency of the signal
		duration : int, optional
			Duration of the sound sample in ms

		"""
		sample = int(duration * 0.001 * self.samplerate)
		time = np.linspace(0, duration * 0.001, num=sample)
		#frequencies =  np.linspace(start_freq, end_freq / 2, num=sample)

		k = (end_freq - start_freq)/ (duration*0.001)
		sweep = (start_freq + k/2 * time) * time
		modulation = self.amplitude * np.sin(2* np.pi * sweep)

		self.signal = np.array(modulation)
		self.freq = {'start_freq': start_freq, 'end_freq': end_freq}

	def amplitude_modulation(self, freq, am_freq, duration=500, ramp=0.01):
		"""Generate an aplitude-modulated tone at a reference frequency

		Parameters
		----------
		freq : int
			Frequency of the signal
		am_freq : int
			Frequency of the amplitude-modulation
		duration : int
			Duration of the soudn sample in ms
		"""
		sample = int(duration * 0.001 * self.samplerate)
		time = np.arange(sample)
		amplitude = 0.5 * (1 - np.cos(2 * np.pi * am_freq * time / self.samplerate))

		if ramp:
			ramp_len = int(self.samplerate * ramp)
			ramp_sample = np.linspace(0, 1, ramp_len)
			amplitude[:ramp_len] = amplitude[:ramp_len] * ramp_sample
			amplitude[-ramp_len:] = amplitude[-ramp_len:] * list(reversed(ramp_sample))

		modulated_signal = [A * self.amplitude * np.sin(2* np.pi * freq * t / self.samplerate) for A, t in zip(amplitude, time)]

		self.signal = np.array(modulated_signal)
		self.freq = {'freq': freq, 'am_freq': am_freq}

	def freq_noise(self, freq, noise_vol, duration=500):
		"""Create a pure tone with noise in the background of increasing intensity

		Parameters
		----------
		freq : int
			Frequency of the signal
		noise_vol : float
			Volume of the white noise in background
		duration : int, optional
			Duration of the sound sample in ms
		"""
		sample = int(duration * 0.001 * self.samplerate)
		time = np.arange(sample)
		noise = noise_vol * np.random.normal(0, 1, len(time))
		noisy_signal = noise + np.array([np.sin(2 * np.pi * freq * t / self.samplerate) for t in time])

		self.signal = np.array(noisy_signal)
		self.freq = {'freq': freq, 'noise_vol': noise_vol}

	def multi_freqs(self, freqs, duration=500):
		""" Generate multiple frequencies sounds

		Parameters
		----------
		freqs : list
			Frequencies of the signal
		duration : int, optional

		"""
		sample = int(duration * 0.001 * self.samplerate)
		time = np.arange(sample)
		all_freqs = np.sum(np.array([[np.sin(2 * np.pi * freq * t / self.samplerate) for t in time] for freq in freqs]), axis=0)
		all_freqs = np.squeeze(normalize(all_freqs[np.newaxis, :], norm='max'))

		self.signal = all_freqs
		self.freq = {'freq{}'.format(i): f for i, f in enumerate(freqs)}

	def harmonics(self, base_freq, patterns, duration=500):
		""" Generate patterns of harmonics

		Parameters
		----------
		base_freq : int
			Fundamental frequency
		patterns : list
			List of integers of relative amplitude of each harmonic. Begins at the first harmonic
		duration : int
		"""
		sample = int(duration * 0.001 * self.samplerate)
		time = np.arange(sample)
		patterns = [1] + patterns
		all_freqs = np.sum(np.array([[a * np.sin(2 * np.pi * base_freq * (i + 1) * t / self.samplerate) for t in time] for i, a in enumerate(patterns)]), axis=0)
		all_freqs = all_freqs / len(patterns)
		print(all_freqs.shape)

		self.signal = all_freqs
		self.freq = {'freq{}'.format(i): base_freq * (i+2) for i, a in enumerate(patterns)}
		self.freq['freq'] = base_freq

	def steps(self, start_freq, end_freq, nstep, spacing='Log', duration=500, ramp=0.01):
		"""Generate a frequency modulated tone in steps

		Parameters
		----------
		start_freq : int
			Starting frequency of the signal
		end_freq : int
			Ending frequency of the signal
		nstep : int
			Number of steps
		spacing : str
			Type of spacing between frequencies. Can be Linear or Log
		duration : int, optional
			Duration of the soudn sample in ms. Default is 500ms
		"""
		sample = int(duration * 0.001 * self.samplerate)
		time = np.arange(sample)

		times = [duration//nstep for i in range(nstep)][:-1]

		times = np.cumsum(times) * 0.001 * self.samplerate
		times = [int(t) for t in times]

		step_times = np.split(time, times)
		
		if spacing == 'Log':
			freq_steps = np.geomspace(start_freq, end_freq, nstep)

		if spacing == 'Linear':
			freq_steps = np.linspace(start_freq, end_freq, nstep)


		tone = self.amplitude * np.sin(2 * np.pi * freq_steps[0] * step_times[0] / self.samplerate)


		for i, freq in enumerate(freq_steps[1:]):


			if tone[-1] < 0:
				for i, v in enumerate(reversed(tone[-int(self.samplerate/(freq)):])):
					if v > 0:
						last_positive = i
						break
				tone = tone[:-(last_positive+ round(self.samplerate/(freq*2)))]
			else:
				for i, v in enumerate(reversed(tone[-int(self.samplerate/(freq)):])):
					if v < 0:
						last_negative = i
						break
				tone = tone[:-last_negative]

			current_tone = self.amplitude * np.sin(2 * np.pi * freq * np.arange(step_times[0][-1]) / self.samplerate)

			tone = np.concatenate((tone, current_tone))

		if ramp:
			ramp_len = int(self.samplerate * ramp)
			ramp_sample = np.linspace(0, 1, ramp_len)
			tone[:ramp_len] = tone[:ramp_len] * ramp_sample
			tone[-ramp_len:] = tone[-ramp_len:] * list(reversed(ramp_sample))

		tone = np.pad(tone, (0, int(duration * 0.001 * self.samplerate) - len(tone)))


		self.signal = tone
		self.freq = {'start_freq': start_freq, 'end_freq': end_freq}


	def save_wav(self, path=None, name=None, bit16=True):
		""" Saves the signal as a .wav file

		Parameters
		----------
		name : str, optional
			Name fo the file to save
		path : str, optional
			Defines a path where to save the wavfile
		bit16 : bool, optional
			Saves the sound as 16-bit integer. For Elphy software. Defaul is True
		"""
		assert self.signal is not None, 'You must define a signal to save'



		if name is None:
			name = self.name

		if bit16:
			signal = np.array(self.signal*32767).astype(np.int16)

		signal = self.signal

		if path is None:
			wavfile.write(os.path.join('../Samples/{}.wav'.format(name)), self.samplerate, signal)
		else:
			wavfile.write(os.path.join(path, name + '.wav'), self.samplerate, signal)

def main():
	parser = argparse.ArgumentParser(description='Parameters for computing')

	parser.add_argument('--inline', '-i', action='store_true',
					    help='for inline generated sounds')
	parser.add_argument('--puretone', '-p', type=int,
						help='Generate a pure tone frequency, please specify frequency in Hz')
	parser.add_argument('--noise', '-n', type=float, nargs=2,
						help='Specify frequency (Hz) and noise(btw 0 and 1')
	parser.add_argument('--ampmod', '-am', type=int, nargs=2,
						help='Amplitude modulation. Base frequency and modulation frequency in Hz')
	parser.add_argument('--harmonic', '-ha', type=int, nargs='*',
	 					help='Generate harmnics Enter frequencies in Hz')
	parser.add_argument('--freqmod', '-fm', type=int, nargs=2,
						help='Ramp frequency generation')
	parser.add_argument('--steps', '-st', type=int, nargs=3,
						help='Steps generation')
	parser.add_argument('--duration', '-d', type=int, default=500,
						help='Duration of the stimulus in ms')
	parser.add_argument('--path', '-a', type=str, default='Samples/',
						help='Path where to save produced stimulus')
	parser.add_argument('--name', '-na', type=str, default=None,
						help='Name of the file generated')

	args = parser.parse_args()

	if args.inline:
		if not os.path.exists(args.path):
			os.makedirs(args.path)

		if args.puretone:
			pure = Sound()
			pure.pure_tone(args.puretone, duration=args.duration)
			pure.save_wav(path=args.path, name=args.name)

		elif args.noise:
			noise = Sound()
			noise.freq_noise(args.noise[0], args.noise[1], duration=args.duration)
			noise.save_wav(path=args.path, name=args.name)

		elif args.ampmod:
			am = Sound()
			am.amplitude_modulation(args.ampmod[0], args.ampmod[1], duration=args.duration)
			am.save_wav(path=args.path, name=args.name)
		
		elif args.steps:
			steps = Sound()
			steps.steps(args.steps[0], args.steps[1], args.steps[2], duration=args.duration)
			steps.save_wav(path=args.path, name=args.name)

		elif args.freqmod:
			freqmod = Sound()
			freqmod.freq_modulation(args.freqmod[0], args.freqmod[1], duration=args.duration)
			freqmod.save_wav(path=args.path, name=args.name)

		elif args.harmonic:
			harmonic = Sound()
			harmonic.multi_freqs(args.harmonic, duration=args.duration)
			harmonic.save_wav(path=args.path, name=args.name)

if __name__=="__main__":
	main()
