from argparse import ArgumentParser, Namespace
from dataclasses import dataclass

@dataclass
class TrainingArgs:
    """Training Arguments"""
    args: Namespace