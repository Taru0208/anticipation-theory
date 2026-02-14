"""Built-in game definitions for the Theory of Anticipation."""

from .coin_toss import CoinToss
from .rps import RPS
from .hpgame import HpGame
from .goldgame import GoldGame
from .coin_duel import CoinDuel
from .coin_duel_rage import CoinDuelRage
from .draft_wars import DraftWars
from .chain_reaction import ChainReaction

__all__ = ["CoinToss", "RPS", "HpGame", "GoldGame", "CoinDuel", "CoinDuelRage", "DraftWars", "ChainReaction"]
