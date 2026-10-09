"""Render a saved DQN episode with the actual Pygame drawer and save it as GIF.

Run with SDL_VIDEODRIVER=dummy on machines without a desktop.
"""
import argparse
from pathlib import Path
import random

import pygame
from PIL import Image
import torch

from src.controller import DQNAgent
from src.drawer import PygameDrawer
from src.pacman_runner import create_environment
from src.state import ActionSpaceEnum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--fps', type=int, default=8)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error('--fps must be positive')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    env = create_environment(**payload['environment'])
    agent = DQNAgent.from_checkpoint(args.checkpoint, env)
    random.seed(args.seed)
    observation = env.reset()
    drawer = PygameDrawer(grid_size=env.grid_size, cell_size=40, framerate=0)
    frames = []
    action = ActionSpaceEnum.RIGHT
    try:
        while True:
            pygame.event.pump()
            drawer.draw(observation.map, action)
            frame = Image.frombytes('RGB', drawer.screen.get_size(),
                                    pygame.image.tostring(drawer.screen, 'RGB'))
            frames.append(frame)
            if observation.done:
                break
            action = agent.get_action(observation)
            observation = env.step(action)
    finally:
        drawer.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], loop=0,
                   duration=[1000 // args.fps] * (len(frames) - 1) + [1000])
    print(f'Saved {len(frames)} frames; score={observation.score}, steps={observation.step_count}: {output}')


if __name__ == '__main__':
    main()
