"""Small original arcade effects, synthesized locally without audio assets."""
from array import array
from io import BytesIO
import math
import wave
import warnings

import pygame


def step_sound(previous_score, observation):
    """Terminal effects take priority over a pellet eaten on the same move."""
    if observation.done:
        if observation.map.pacman_position in observation.map.ghost_positions:
            return 'death'
        return 'win' if not observation.map.pellets else 'timeout'
    return 'eat' if observation.score > previous_score else None


def tone_sequence(notes, sample_rate=22050):
    samples = array('h')
    for frequency, duration in notes:
        count = int(sample_rate * duration)
        for index in range(count):
            # Short fades prevent clicks at the edges of each note.
            envelope = min(1.0, index / (sample_rate * .005),
                           (count - 1 - index) / (sample_rate * .012))
            phase = 2 * math.pi * frequency * index / sample_rate
            value = math.sin(phase) + .2 * math.sin(2 * phase)
            samples.append(int(10000 * envelope * value))
    import sys
    if sys.byteorder != 'little':
        samples.byteswap()
    output = BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    output.seek(0)
    return output


class GameAudio:
    EFFECTS = {
        'eat': ((660, .035), (440, .045)),
        'win': ((523, .09), (659, .09), (784, .09), (1047, .18)),
        'death': ((440, .09), (330, .09), (220, .12), (110, .16)),
        'timeout': ((330, .12), (262, .16)),
    }

    def __init__(self, enabled=True, volume=.25):
        if not 0 <= volume <= 1:
            raise ValueError('sound_volume must be between 0 and 1')
        self.enabled = enabled
        self.volume = volume
        self.channel = None
        self.sounds = {}
        self.failed = False
        if enabled:
            self._initialize()

    def _initialize(self):
        if self.channel is not None or self.failed:
            return
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self.sounds = {
                name: pygame.mixer.Sound(file=tone_sequence(notes))
                for name, notes in self.EFFECTS.items()
            }
            self.channel = pygame.mixer.Channel(0)
            self.channel.set_volume(self.volume)
        except pygame.error as error:
            self.enabled = False
            self.failed = True
            warnings.warn(f'Game audio unavailable; continuing silently: {error}', RuntimeWarning)

    def toggle(self):
        self.enabled = not self.enabled
        if self.enabled:
            self._initialize()
            if self.failed:
                self.enabled = False
        elif self.channel is not None:
            self.channel.stop()

    def play(self, name):
        if self.enabled and self.channel is not None and name is not None:
            self.channel.play(self.sounds[name])

    def busy(self):
        return self.enabled and self.channel is not None and self.channel.get_busy()

    @property
    def status(self):
        return 'unavailable' if self.failed else ('on' if self.enabled else 'off')
