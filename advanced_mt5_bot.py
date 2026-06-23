import datetime as dt
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pandas_ta as ta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


@dataclass(frozen=True)
class BotConfig:
    login_id: int = int(os.getenv("MT5_LOGIN", "0"))
    password: str = os.getenv("MT5_PASSWORD", "")
    server: str = os.getenv("MT5_SERVER", "")
    symbol: str = os.getenv("MT5_SYMBOL", "XAUUSDm")
    timeframe: int = mt5.TIMEFRAME_M5
    bars: int = int(os.getenv("MT5_BARS", "1500"))
    poll_seconds: int = int(os.getenv("BOT_POLL_SECONDS", "15"))
    risk_per_trade: float = float(os.getenv("BOT_RISK_PER_TRADE", "0.01"))
    max_lot: float = float(os.getenv("BOT_MAX_LOT", "0.50"))
    min_lot: float = float(os.getenv("BOT_MIN_LOT", "0.01"))
    model_path: str = os.getenv("BOT_MODEL_PATH", "super_brain_v3")
    magic: int = int(os.getenv("BOT_MAGIC", "2026"))
    retrain_hour: int = int(os.getenv("BOT_RETRAIN_HOUR", "23"))
    retrain_minute: int = int(os.getenv("BOT_RETRAIN_MINUTE", "45"))
    retrain_steps: int = int(os.getenv("BOT_RETRAIN_STEPS", "6000"))


class AdvancedBrainEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.features = [
            "close",
            "ema_10",
            "ema_50",
            "ema_200",
            "rsi",
            "atr",
            "spread",
            "adx",
            "returns",
            "volatility",
        ]
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)  # 0=hold,1=buy,2=sell
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.features),),
            dtype=np.float32,
        )

    def _state(self) -> np.ndarray:
        return self.df.loc[self.current_step, self.features].values.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._state(), {}

    def step(self, action: int):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        row = self.df.iloc[self.current_step]
        prev = self.df.iloc[self.current_step - 1]
        diff = row["close"] - prev["close"]

        trade_cost = row["spread"] * 0.1
        momentum_bonus = abs(row["returns"]) * 20

        if action == 1:  # buy
            reward = (diff * 100) - trade_cost
        elif action == 2:  # sell
            reward = (-diff * 100) - trade_cost
        else:  # hold
            reward = -0.03

        if action != 0 and row["adx"] > 20:
            reward += momentum_bonus

        return self._state(), float(reward), done, False, {}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def initialize_mt5(cfg: BotConfig) -> None:
    if cfg.login_id <= 0 or not cfg.password or not cfg.server:
        raise ValueError("Set MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER environment variables.")

    if not mt5.initialize(login=cfg.login_id, password=cfg.password, server=cfg.server):
        error = mt5.last_error()
        raise RuntimeError(f"MT5 initialization failed: {error}")

    logging.info("Connected to MT5 account %s on %s", cfg.login_id, cfg.server)


def fetch_data(cfg: BotConfig) -> Optional[pd.DataFrame]:
    rates = mt5.copy_rates_from_pos(cfg.symbol, cfg.timeframe, 0, cfg.bars)
    if rates is None:
        return None

    df = pd.DataFrame(rates)
    df["ema_10"] = ta.ema(df["close"], length=10)
    df["ema_50"] = ta.ema(df["close"], length=50)
    df["ema_200"] = ta.ema(df["close"], length=200)
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["adx"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
    df["returns"] = df["close"].pct_change().fillna(0.0)
    df["volatility"] = df["returns"].rolling(20).std().fillna(0.0)
    df.dropna(inplace=True)
    return df


def calculate_lot(balance: float, atr: float, cfg: BotConfig) -> float:
    if atr <= 0:
        return cfg.min_lot
    risk_amount = balance * cfg.risk_per_trade
    stop_distance = atr * 2.0
    raw_lot = risk_amount / (stop_distance * 100)
    return round(max(cfg.min_lot, min(raw_lot, cfg.max_lot)), 2)


def trend_filter(last: pd.Series, action: int) -> bool:
    buy_ok = action == 1 and last["close"] > last["ema_200"] and last["rsi"] > 55 and last["adx"] > 18
    sell_ok = action == 2 and last["close"] < last["ema_200"] and last["rsi"] < 45 and last["adx"] > 18
    return buy_ok or sell_ok


def send_order(cfg: BotConfig, last: pd.Series, action: int) -> None:
    tick = mt5.symbol_info_tick(cfg.symbol)
    acc = mt5.account_info()
    if tick is None or acc is None:
        logging.warning("Cannot place order: missing tick/account info")
        return

    is_buy = action == 1
    price = tick.ask if is_buy else tick.bid
    sl_distance = last["atr"] * 2.0
    lot = calculate_lot(acc.balance, last["atr"], cfg)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": cfg.symbol,
        "volume": float(lot),
        "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
        "price": float(price),
        "sl": float(price - sl_distance if is_buy else price + sl_distance),
        "tp": float(price + sl_distance * 3 if is_buy else price - sl_distance * 3),
        "magic": cfg.magic,
        "comment": "Advanced AI",
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(request)
    if result is None:
        logging.error("Order send failed: %s", mt5.last_error())
        return
    logging.info("Order result | retcode=%s | order=%s", result.retcode, result.order)


def manage_positions(cfg: BotConfig, last_atr: float) -> None:
    positions = mt5.positions_get(symbol=cfg.symbol)
    if not positions:
        return

    for pos in positions:
        is_buy = pos.type == mt5.POSITION_TYPE_BUY
        entry = pos.price_open
        current = pos.price_current
        sl = pos.sl
        profit_dist = abs(current - entry)

        if profit_dist > last_atr * 1.5:
            breakeven = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": pos.ticket,
                "sl": float(entry),
                "tp": float(pos.tp),
            }
            mt5.order_send(breakeven)

        trail_gap = last_atr * 1.5
        if is_buy:
            new_sl = current - trail_gap
            if new_sl > sl:
                mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "sl": float(new_sl), "tp": float(pos.tp)})
        else:
            new_sl = current + trail_gap
            if sl == 0 or new_sl < sl:
                mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "sl": float(new_sl), "tp": float(pos.tp)})


def load_or_create_model(df: pd.DataFrame, cfg: BotConfig) -> PPO:
    env = DummyVecEnv([lambda: AdvancedBrainEnv(df)])
    if os.path.exists(f"{cfg.model_path}.zip"):
        logging.info("Loading model from %s", cfg.model_path)
        return PPO.load(cfg.model_path, env=env)

    logging.info("Creating a fresh PPO model")
    return PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=512,
        batch_size=128,
    )


def maybe_retrain(model: PPO, now: dt.datetime, last_trained_day: Optional[int], cfg: BotConfig) -> Optional[int]:
    if now.hour == cfg.retrain_hour and now.minute >= cfg.retrain_minute and last_trained_day != now.day:
        logging.info("Daily retraining for %s timesteps", cfg.retrain_steps)
        model.learn(total_timesteps=cfg.retrain_steps)
        model.save(cfg.model_path)
        return now.day
    return last_trained_day


def run_bot() -> None:
    setup_logging()
    cfg = BotConfig()
    initialize_mt5(cfg)

    df_init = fetch_data(cfg)
    if df_init is None or df_init.empty:
        raise RuntimeError("No market data returned from MT5")

    model = load_or_create_model(df_init, cfg)
    last_trained_day: Optional[int] = None

    try:
        while True:
            df = fetch_data(cfg)
            if df is None or df.empty:
                logging.warning("No data received, retrying")
                time.sleep(cfg.poll_seconds)
                continue

            last = df.iloc[-1]
            state = last[
                ["close", "ema_10", "ema_50", "ema_200", "rsi", "atr", "spread", "adx", "returns", "volatility"]
            ].values.astype(np.float32)

            action, _ = model.predict(state, deterministic=True)
            action = int(action)
            positions = mt5.positions_get(symbol=cfg.symbol)

            if not positions and action in (1, 2) and trend_filter(last, action):
                send_order(cfg, last, action)

            manage_positions(cfg, float(last["atr"]))
            last_trained_day = maybe_retrain(model, dt.datetime.now(), last_trained_day, cfg)
            time.sleep(cfg.poll_seconds)

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        mt5.shutdown()
        logging.info("MT5 connection closed")


if __name__ == "__main__":
    run_bot()
