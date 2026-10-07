from src.environment import GhostsPacmanEnvironment


def test_from_map_preserves_original():
    # Build a fresh environment and play a few steps.
    env = GhostsPacmanEnvironment(grid_size=8, num_ghosts=2, max_steps=100)
    env.reset()

    for _ in range(5):
        env.random_step()

    # Save a snapshot of the original map.
    original_map = env.map.clone()

    # Build a simulation environment from the current state.
    sim_env = GhostsPacmanEnvironment.from_map(env.map, past_steps=env.step_count)

    # Verify the simulation environment's map matches the original.
    assert sim_env.map.pacman_position == original_map.pacman_position
    assert sim_env.map.pellets == original_map.pellets
    assert sim_env.map.ghost_positions == original_map.ghost_positions
    assert sim_env.map.walls == original_map.walls
    assert sim_env.step_count == env.step_count

    # Step the simulation a few times.
    from src.state import ActionSpaceEnum
    for _ in range(10):
        sim_env.random_step()

    # Verify the original environment's map is untouched.
    assert env.map.pacman_position == original_map.pacman_position
    assert env.map.pellets == original_map.pellets
    assert env.map.ghost_positions == original_map.ghost_positions
    assert env.map.walls == original_map.walls

    print("from_map preserves original test passed.")


def test_clone_is_independent():
    env = GhostsPacmanEnvironment(grid_size=8, num_ghosts=2, max_steps=100)
    env.reset()
    a = env.map
    b = a.clone()

    a_walls_len = len(a.walls)
    a_pellets_len = len(a.pellets)
    a_ghosts = set(a.ghost_positions)

    from src.state import Position
    b.walls.add(Position(-1, -1))          # not in a
    if b.pellets:
        b.pellets.discard(next(iter(b.pellets)))
    b.ghost_positions.add(Position(99, 99))  # not in a

    assert len(a.walls) == a_walls_len
    assert len(a.pellets) == a_pellets_len
    assert a.ghost_positions == a_ghosts

    print("Clone independence test passed.")


if __name__ == "__main__":
    test_from_map_preserves_original()
    test_clone_is_independent()