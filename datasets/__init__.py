''' It is useful to keep audio and its annotation together because we can apply
transformations such as concatenation of training examples. '''
class AnnotatedAudio:
	def __init__(self, audio, annotation):
		self.audio = audio
		self.annotation = annotation

''' Audio container
Audio can be represented in different forms - as raw audio with various
sampling rates or as spectrograms. '''
class Audio:
	def __init__(self, path, uid):
		self.uid = uid
		self.path = path

	def get_resampled_audio(self, samplerate):
		pass

	def get_spectrogram(self):
		pass

''' Handles the common time-frequency annotation format. '''
class Annotation:
	def __init__(self, times, frequencies):
		self.times = np.array(times)
		self.frequencies = np.array(frequencies)

	def get_multif0(self):
		return self.times, self.frequencies

	def get_notes(self):
		return self.times, self.notes

class Dataset:
	def __init__(self, annotated_audios):
		self.annotated_audios = annotated_audios

	def get_annotated_audio_windows(self, annotations_per_window, context_width):
		pass