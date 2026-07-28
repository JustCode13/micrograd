import logging
import os
import random
import typing
import numpy

from ..logging.logger import ProjectLogger
from ..exceptions.errors import RandomSeedError


class SeedManager:
    def __init__(self, seed: int) -> None:

        if seed < 0:
            raise ValueError("seed cannot be negative")

        self.seed = seed
        self.logger = ProjectLogger(logger_name="SeedManager")

        self.logger.info(f"SeedManager initialized with seed: {seed}")
