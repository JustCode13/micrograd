import logging
import os
import random
import typing
import numpy as np

from logger.logger import ProjectLogger
from exceptions.errors import RandomSeedError


class SeedManager:
    def __init__(self, seed: int) -> None:

        if seed < 0:
            raise ValueError("seed cannot be negative")

        self.seed = seed
        self.logger = ProjectLogger(logger_name="SeedManager")
        self.logger.configure()

        self.logger.info(f"SeedManager initialized with seed: {seed}")

    def apply(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)

        os.environ["PYTHONHASHSEED"] = str(self.seed)

        self.logger.info(f"Random generators initialized with seed {self.seed}")

    def random_generator(self) -> random.Random:
        generator = random.Random(self.seed)
        return generator

    def export(self) -> dict[str, int]:
        snapshot = {
            "seed": self.seed,
        }

        return snapshot
