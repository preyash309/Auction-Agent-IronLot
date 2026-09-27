import argparse
import json

from .agent import LiveAuctionAgent


def main(argv=None):
    parser = argparse.ArgumentParser(description="Used-car valuation and risk-aware auction research")
    subs = parser.add_subparsers(dest="command", required=True)
    predict = subs.add_parser("predict", help="Analyze one vehicle JSON and suggest the next bid")
    predict.add_argument("--item", required=True)
    predict.add_argument("--checkpoints")
    predict.add_argument("--current-bid", type=float, default=0)
    training = subs.add_parser("train", help="Train and evaluate a new seeded 80/20 run")
    training.add_argument("--data", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--baseline", action="store_true")
    training.add_argument("--estimators", type=int, help="Override both tree counts for smoke runs")
    evaluation = subs.add_parser("evaluate", help="Price metrics on caller-supplied labeled data")
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--checkpoints")
    simulation = subs.add_parser("simulate", help="Seeded synthetic-opponent tournament")
    simulation.add_argument("--data", required=True)
    simulation.add_argument("--checkpoints")
    simulation.add_argument("--rounds", type=int, default=500)
    simulation.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        if args.command == "predict":
            with open(args.item, encoding="utf-8") as stream:
                item = json.load(stream)
            agent = LiveAuctionAgent(args.checkpoints)
            agent.analyze_item(item)
            result = {"predicted_price": agent.predicted_price, "p10_price": agent.p10_price, "next_bid": agent.place_bid(args.current_bid)}
        elif args.command == "train":
            from .training import train
            if args.estimators is not None and args.estimators <= 0:
                raise ValueError("Estimators must be positive")
            result = train(args.data, args.output, args.seed, args.baseline, args.estimators)
        elif args.command == "evaluate":
            from .training import read_dataset, prepare, metrics
            agent = LiveAuctionAgent(args.checkpoints)
            frame = read_dataset(args.data)
            result = metrics(frame["sellingprice"], agent.model.predict(prepare(frame, agent.others)))
            result["protocol"] = "caller-supplied data; no guarantee of separation from training"
        else:
            from .simulation import simulate
            result = simulate(args.data, args.checkpoints, args.rounds, args.seed)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")
