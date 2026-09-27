from collections import deque
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

AVAILABLE = bool(importlib.util.find_spec("gymnasium") and importlib.util.find_spec("optuna"))
if AVAILABLE:
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env
    from auction_bot.environment import CarAuctionEnv
    from auction_bot.optimize import historical_max_bid, optimize, score_policy

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(AVAILABLE, "Install .[agent-training] to test IronLot environment and optimization")
class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"pred": [11000.0]*12, "std": [1000.0]*12, "sellingprice": [10000.0+i*50 for i in range(12)]})

    def test_notebook_economic_transition_matches(self):
        notebook = json.loads((ROOT/"PreyashPratyush_IronLot/environment_Preyash_Pratyush.ipynb").read_text())
        source = next("".join(cell["source"]) for cell in notebook["cells"] if "class CarAuctionEnv(" in "".join(cell.get("source",[])))
        scope = {"gym": gym, "np": np, "deque": deque}
        exec(compile(source,"historical_environment_cell","exec"),scope)
        original = scope["CarAuctionEnv"](self.frame)
        original.reset(); original.dataset = self.frame
        adapted = CarAuctionEnv(self.frame,rounds=3)
        adapted.reset(seed=42); adapted.dataset = self.frame
        with patch("numpy.random.normal",return_value=0.0), patch.object(adapted,"_np_random",SimpleNamespace(normal=lambda *args:0.0)):
            for bid in (12000.0,500.0):
                actual = adapted.step(bid); expected = original.step(bid)
                np.testing.assert_array_equal(actual[0],expected[0])
                self.assertEqual(actual[1:],expected[1:])

    def test_terminal_observation_and_reset(self):
        env = CarAuctionEnv(self.frame.iloc[:1],rounds=1)
        env.reset(seed=4)
        observation, reward, terminated, truncated, info = env.step(0.0)
        self.assertTrue(terminated); self.assertFalse(truncated)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(observation[0],0)
        with self.assertRaises(RuntimeError): env.step(0.0)
        env.reset(seed=4)
        self.assertEqual(env.current_car_index,0)

    def test_seed_and_gymnasium_contract(self):
        env = CarAuctionEnv(self.frame,rounds=5)
        first,_ = env.reset(seed=123); first_step = env.step(np.array([11000.0]))
        second,_ = env.reset(seed=123); second_step = env.step(np.array([11000.0]))
        np.testing.assert_array_equal(first,second)
        np.testing.assert_array_equal(first_step[0],second_step[0])
        self.assertEqual(first_step[1:],second_step[1:])
        check_env(env,skip_render_check=True)

    def test_historical_objective_bid_formula(self):
        notebook = json.loads((ROOT/"PreyashPratyush_IronLot/environment_Preyash_Pratyush.ipynb").read_text())
        source = next("".join(cell["source"]) for cell in notebook["cells"] if "def objective(trial):" in "".join(cell.get("source",[])))
        scope={"np":np}
        exec(compile(source,"historical_objective","exec"),scope)
        expected=[]
        class FakeEnv:
            def __init__(self,frame): self.bankroll=500000
            def reset(self): return np.array([15000,3000,600000,1,0.1],dtype=np.float32),{}
            def step(self,bid): expected.append(bid); return None,0,True,False,{}
        class Trial:
            def suggest_float(self,name,*args): return {"alpha":0.1007,"beta":0.3792,"delta":0.0125}[name]
        scope.update(CarAuctionEnv=FakeEnv,df=self.frame)
        scope["objective"](Trial())
        actual=historical_max_bid([15000,3000,600000,1,float(np.float32(0.1))],0.1007,0.3792,0.0125)
        self.assertAlmostEqual(actual,expected[0],places=10)
        self.assertEqual(historical_max_bid([1,100000,1,1,0],1.5,0.5,0),0)

    def test_bankruptcy_matches_original_penalty(self):
        env=CarAuctionEnv(self.frame,rounds=3); env.reset(seed=1); env.bankroll=100
        with patch.object(env,"_np_random",SimpleNamespace(normal=lambda *args:0.0)):
            _,reward,done,_,info=env.step(20000)
        self.assertEqual(reward,-100); self.assertTrue(done)
        self.assertFalse(info["won"]); self.assertEqual(env.bankroll,0)

    def test_dataset_and_action_validation(self):
        with self.assertRaises(ValueError): CarAuctionEnv(self.frame.drop(columns="std"),rounds=1)
        with self.assertRaises(ValueError): CarAuctionEnv(self.frame,rounds=20)
        invalid=self.frame.copy(); invalid.loc[0,"std"]=np.nan
        with self.assertRaises(ValueError): CarAuctionEnv(invalid,rounds=1)
        env=CarAuctionEnv(self.frame,rounds=1); env.reset(seed=2)
        with self.assertRaises(ValueError): env.step(float("nan"))
        with self.assertRaises(ValueError): env.step([1,2])

    def test_optimization_persists_study_and_is_seeded(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset=Path(directory)/"predictions.csv"; self.frame.to_csv(dataset,index=False)
            first=optimize(dataset,Path(directory)/"first",trials=2,games=2,rounds=3,seed=42)
            second=optimize(dataset,Path(directory)/"second",trials=2,games=2,rounds=3,seed=42)
            self.assertEqual(first,second)
            self.assertTrue((Path(directory)/"first/study.sqlite3").is_file())
            self.assertTrue((Path(directory)/"first/trials.csv").is_file())
            self.assertEqual(score_policy(self.frame,first["best_parameters"],rounds=3,games=2,seed=42),first["mean_final_bankroll"])
            with self.assertRaises(ValueError): optimize(dataset,Path(directory)/"first",trials=1,games=1,rounds=3)


if __name__ == "__main__": unittest.main()
