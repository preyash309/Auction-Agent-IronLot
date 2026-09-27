import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from auction_bot.agent import ASSET_NAMES, LiveAuctionAgent
from auction_bot.preprocessing import finalizeData
from auction_bot.training import fit_metadata, make_pipeline, prepare, train
from auction_bot.simulation import simulate

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("historical", ROOT/"experiments/archive/submitted_agent.py")
historical = importlib.util.module_from_spec(spec)
spec.loader.exec_module(historical)
ITEM = json.loads((ROOT/"examples/car.json").read_text())
METADATA = {"cleanup_names": {"ford truck": "ford"}, "trim_map": {"Ford": "xlt"}, "body_map": {"Ford": "sedan"}, "median": 60000.0}


class FixedModel:
    def __init__(self, value): self.value = value
    def predict(self, frame): return np.array([self.value]*len(frame))


def fake_agent():
    with tempfile.TemporaryDirectory() as directory:
        for name in ASSET_NAMES: (Path(directory)/name).touch()
        with patch("auction_bot.agent.pickle.load", side_effect=[FixedModel(15000), FixedModel(12000), METADATA]):
            return LiveAuctionAgent(directory)


class AgentTests(unittest.TestCase):
    def test_preprocessing_matches_submission(self):
        rows = [ITEM, {**ITEM, "make": "ford truck", "trim": None, "body": None, "condition": None, "odometer": None}, {**ITEM, "year": 2016, "make": "lamborghini"}, {**ITEM, "make": None, "model": None}]
        args = [METADATA[k] for k in ("cleanup_names", "trim_map", "body_map", "median")]
        actual = finalizeData(pd.DataFrame(rows), *args)
        expected = historical.finalizeData(pd.DataFrame(rows), *args)
        assert_frame_equal(actual, expected)
        self.assertEqual(actual.loc[2,"is_rare"], 1)
        self.assertTrue(np.isinf(actual.loc[2,"miles_per_year"]))

    def test_bid_matches_submission_across_states(self):
        agent = fake_agent(); agent.analyze_item(ITEM)
        original = historical.LiveAuctionAgent.__new__(historical.LiveAuctionAgent)
        for bankroll in (0, 5000, 250000, 500000, 600000):
            for rate in (0, 0.1, 0.2, 0.8):
                for current in (0, 10000, 14650, 20000):
                    agent.bankroll = bankroll; agent.win_rate = rate
                    original.__dict__.update(agent.__dict__)
                    self.assertEqual(agent.place_bid(current), original.place_bid(current))

    def test_no_stale_bid_after_invalid_item(self):
        agent = fake_agent(); agent.analyze_item(ITEM)
        self.assertGreater(agent.place_bid(100), 0)
        with self.assertRaises(ValueError): agent.analyze_item({})
        self.assertEqual(agent.place_bid(100), 0)

    def test_missing_identity_folds(self):
        agent = fake_agent(); agent.analyze_item({**ITEM, "make": None})
        self.assertEqual(agent.predicted_price, 0)
        self.assertEqual(agent.place_bid(0), 0)

    def test_bankroll_and_results(self):
        agent = fake_agent()
        self.assertEqual(agent.place_bid(0), 0)
        agent.auction_result(True, 1000, 1200, 500200)
        agent.auction_result(False, 1000, 1200, 500200)
        self.assertEqual(agent.win_rate, 0.5)
        self.assertEqual(agent.bankroll, 500200)
        with self.assertRaises(ValueError): agent.place_bid(float("nan"))
        with self.assertRaises(ValueError): agent.auction_result(True, 0, 0, -1)

    def test_missing_checkpoints_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Missing checkpoints"):
                LiveAuctionAgent(directory)

    def test_independent_encoders_and_train_only_metadata(self):
        first = make_pipeline(42); second = make_pipeline(42, quantile=True)
        self.assertIsNot(first.named_steps["preprocessor"], second.named_steps["preprocessor"])
        frame = pd.DataFrame([ITEM, {**ITEM,"odometer": 10000}])
        metadata = fit_metadata(frame)
        self.assertEqual(metadata["median"], (53070+10000)/2)
        self.assertEqual(prepare(frame,metadata).shape[0],2)

    def test_small_training_split_and_reload(self):
        rows = [{**ITEM, "year": 2010+i%5, "odometer": 10000+i*1000, "sellingprice": 5000+i*100} for i in range(40)]
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)/"data.csv"; pd.DataFrame(rows).to_csv(dataset,index=False)
            output = Path(directory)/"run"
            result = train(dataset, output, estimators=2)
            split = json.loads((output/"split.json").read_text())
            self.assertFalse(set(split["train"]) & set(split["test"]))
            self.assertEqual(len(split["train"])+len(split["test"]),40)
            self.assertEqual(result["rows"],8)
            agent = LiveAuctionAgent(output); agent.analyze_item(ITEM)
            self.assertTrue(np.isfinite(agent.predicted_price))
            with self.assertRaisesRegex(ValueError,"empty directory"):
                train(dataset,output,estimators=2)

    def test_simulation_seed_counts_and_validation(self):
        rows = [{**ITEM, "sellingprice": 15000+i*50} for i in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"cars.csv"; pd.DataFrame(rows).to_csv(path,index=False)
            with patch("auction_bot.simulation.LiveAuctionAgent", side_effect=lambda _: fake_agent()):
                first = simulate(path, directory, rounds=10, seed=42)
                second = simulate(path, directory, rounds=10, seed=42)
                self.assertEqual(first,second)
                self.assertEqual(first["auctioned"]+first["skipped"],10)
            with self.assertRaisesRegex(ValueError,"positive"):
                simulate(path,directory,rounds=0)
            with self.assertRaisesRegex(ValueError,"labeled rows"):
                simulate(path,directory,rounds=20)


@unittest.skipUnless(os.environ.get("AUCTION_TEST_CHECKPOINTS"), "Local checkpoints are optional and excluded from Git")
class CheckpointTests(unittest.TestCase):
    def test_real_checkpoint_prediction_and_bid_equivalence(self):
        directory = Path(os.environ["AUCTION_TEST_CHECKPOINTS"])
        actual = LiveAuctionAgent(directory)
        original = historical.LiveAuctionAgent.__new__(historical.LiveAuctionAgent)
        original.__dict__.update(actual.__dict__)
        for item in (ITEM, {**ITEM,"make":"lamborghini"}, {**ITEM,"year":2016}, {**ITEM,"condition":None,"odometer":None}, {**ITEM,"make":None}):
            actual.analyze_item(item); original.analyze_item(item)
            self.assertEqual(actual.predicted_price,original.predicted_price)
            self.assertEqual(actual.p10_price,original.p10_price)
            for current in (0,10000,50000):
                self.assertEqual(actual.place_bid(current),original.place_bid(current))


if __name__ == "__main__": unittest.main()
