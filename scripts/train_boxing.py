"""Train (or evaluate) a boxing-tactics policy with PPO.

    pip install -e '.[rl]'
    python scripts/train_boxing.py --timesteps 200000 --n-envs 6          # laptop CPU, a few hours
    python scripts/train_boxing.py --eval build/rl/boxing_ppo.zip --episodes 5
    python scripts/train_boxing.py --random --episodes 3                    # baseline: random macros

The policy only chooses tactics (robotcode.rl.boxing_env); balance and punch execution remain the
model-based controllers, so this runs on CPU. See docs/HUMANOID_BOXING.md for moving training of
low-level skills to an NVIDIA GPU (MuJoCo MJX).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def make_env(rank: int, privileged: bool):
    def _init():
        from robotcode.rl.boxing_env import BoxingEnv
        env = BoxingEnv(privileged_opponent=privileged)
        env.reset(seed=rank)
        return env
    return _init


def evaluate(model, episodes: int, privileged: bool) -> None:
    from robotcode.rl.boxing_env import BoxingEnv
    env = BoxingEnv(privileged_opponent=privileged)
    for ep in range(episodes):
        obs, _ = env.reset(seed=1000 + ep)
        total, done, info = 0.0, False, {}
        while not done:
            action = model.predict(obs, deterministic=True)[0] if model is not None else env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        print(f"episode {ep}: return {total:7.2f}  score {info.get('score')}  time {info.get('time', 0):.1f} s")
    env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--camera", action="store_true", help="opponent from the head camera (slower) instead of ground truth")
    parser.add_argument("--out", default=str(ROOT / "build" / "rl" / "boxing_ppo"))
    parser.add_argument("--eval", help="evaluate a saved model instead of training")
    parser.add_argument("--random", action="store_true", help="evaluate random macros (baseline)")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    privileged = not args.camera
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError:
        parser.error("needs stable-baselines3: pip install -e '.[rl]'")
    if args.random or args.eval:
        evaluate(PPO.load(args.eval) if args.eval else None, args.episodes, privileged)
        return 0
    # "spawn": each worker gets its own process (the opponent renders its camera; OpenGL does not survive fork).
    env = SubprocVecEnv([make_env(i, privileged) for i in range(args.n_envs)], start_method="spawn")
    try:
        import tensorboard  # noqa: F401
        tb = str(ROOT / "build" / "rl" / "tb")
    except ImportError:
        tb = None
    model = PPO("MlpPolicy", env, n_steps=256, batch_size=256, learning_rate=3e-4, gamma=0.97, ent_coef=0.01,
                verbose=1, tensorboard_log=tb)
    model.learn(total_timesteps=args.timesteps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    env.close()
    print(f"saved {args.out}.zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
