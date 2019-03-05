''' It is useful to keep audio and its annotation together because we can apply
transformations such as concatenation of training examples. '''
class AnnotatedAudio:
    def __init__(self, annotation, audio):
        self.audio = audio
        self.annotation = annotation

    def slice(self, start, end):
        return AnnotatedAudio(self.annotation.slice(start, end), self.audio.slice(start, end))
