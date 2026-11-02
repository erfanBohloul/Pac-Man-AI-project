"""Check audio event priority and optional mixer behavior without real speakers."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pygame

from src.drawer.audio import GameAudio, step_sound


class AudioTests(unittest.TestCase):
    def tearDown(self):
        pygame.mixer.quit()

    def test_terminal_events_override_food_on_the_same_step(self):
        state = SimpleNamespace(pacman_position=(1, 1), ghost_positions=set(), pellets={(2, 2)})
        obs = SimpleNamespace(map=state, score=10, done=False)
        self.assertEqual(step_sound(0, obs), 'eat')
        self.assertIsNone(step_sound(10, obs))
        obs.done = True
        self.assertEqual(step_sound(0, obs), 'timeout')
        state.pellets.clear()
        self.assertEqual(step_sound(0, obs), 'win')
        state.ghost_positions.add((1, 1))
        self.assertEqual(step_sound(0, obs), 'death')

    def test_disabled_audio_does_not_initialize_mixer(self):
        with patch('pygame.mixer.init') as initialize:
            audio = GameAudio(enabled=False)
            audio.play('eat')
            initialize.assert_not_called()
        self.assertFalse(audio.busy())

    def test_missing_audio_device_is_nonfatal(self):
        with patch('pygame.mixer.get_init', return_value=None), patch(
                'pygame.mixer.init', side_effect=pygame.error('no device')):
            with self.assertWarnsRegex(RuntimeWarning, 'continuing silently'):
                audio = GameAudio()
        audio.play('death')
        audio.toggle()
        self.assertEqual(audio.status, 'unavailable')
        self.assertFalse(audio.busy())

    def test_effects_decode_and_mute_stops_playback(self):
        with patch.dict(os.environ, {'SDL_AUDIODRIVER': 'dummy'}):
            audio = GameAudio()
            for name, sound in audio.sounds.items():
                self.assertGreater(sound.get_length(), 0)
                audio.play(name)
                self.assertTrue(audio.busy())
            audio.toggle()
            self.assertFalse(audio.busy())
            self.assertFalse(audio.channel.get_busy())
            audio.toggle()
            audio.play('eat')
            self.assertTrue(audio.busy())


if __name__ == '__main__':
    unittest.main()
