import argparse
import json
import signal
from pathlib import Path

from .config import Config
from .network import Network
from .track import TRACKS


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(
        description="BOATBRAIN: ternary driving laboratory"
    )
    sub = parser.add_subparsers(dest="command")
    play = sub.add_parser(
        "play", help="Pygame lab; no model starts with a random brain"
    )
    play.add_argument("--model")
    play.add_argument("--track", choices=TRACKS, default="oval")
    play.add_argument("--run-dir", default="runs/lab")
    play.add_argument("--frames", type=positive, help=argparse.SUPPRESS)
    play.add_argument("--screenshot", help=argparse.SUPPRESS)
    train = sub.add_parser("train", help="headless genetic reinforcement learning")
    train.add_argument("--generations", type=positive, default=60)
    train.add_argument("--population", type=positive, default=64)
    train.add_argument("--steps", type=positive, default=900)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--hidden", type=positive, default=4)
    train.add_argument("--out", default="runs/train")
    train.add_argument(
        "--tracks", nargs="+", choices=TRACKS, help="Training maps; default oval harbor"
    )
    source = train.add_mutually_exclusive_group()
    source.add_argument("--resume")
    source.add_argument("--warm")
    evaluate = sub.add_parser(
        "evaluate", help="nine starts, including held-out track and reverse direction"
    )
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--steps", type=positive, default=1800)
    evaluate.add_argument("--out")
    evaluate.add_argument(
        "--tracks", nargs="+", choices=TRACKS, default=["oval", "harbor", "technical"]
    )
    export = sub.add_parser("export", help="export weights, sensors and gate DAG")
    export.add_argument("--model", required=True)
    export.add_argument("--out", default="runs/wiring")
    args = parser.parse_args()
    if args.command in (None, "play"):
        from .ui import Lab

        if args.command is None:
            Lab().run()
        else:
            Lab(args.model, args.track, args.run_dir).run(args.frames, args.screenshot)
    elif args.command == "train":
        from .training import Evolution

        warm = Network.load(args.warm) if args.warm else None
        config = warm.config if warm else Config(hidden=args.hidden)
        evolution = (
            Evolution.resume(args.resume)
            if args.resume
            else Evolution(
                args.population,
                args.seed,
                config,
                warm,
                tracks=args.tracks or ("oval", "harbor"),
            )
        )
        if args.resume and args.tracks and tuple(args.tracks) != evolution.track_names:
            parser.error(
                "Resume uses checkpoint maps; use --warm for a different map set"
            )
        stop = False

        def request_stop(*_):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        print(
            f"Training {len(evolution.population)} genomes, {evolution.config.parameters} parameters",
            flush=True,
        )
        try:
            for _ in range(args.generations):
                if stop:
                    break
                row = evolution.advance(args.steps)
                evolution.save(args.out)
                print(
                    f"gen {row['generation']:4d}  best {row['best']:+.4f}  mean {row['mean']:+.4f}  all-time {row['champion']:+.4f}  wires {row['nonzero']}",
                    flush=True,
                )
        finally:
            evolution.save(args.out)
    elif args.command == "evaluate":
        from .training import evaluate

        results = evaluate(Network.load(args.model), args.steps, args.tracks)
        output = json.dumps(results, indent=2)
        print(output)
        if args.out:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(output + "\n")
    elif args.command == "export":
        gates = Network.load(args.model).export(args.out)
        print(f"Exported {gates} gates to {args.out}")


if __name__ == "__main__":
    main()
