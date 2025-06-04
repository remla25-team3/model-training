# pylint_custom/ml_pylint.py

from pylint.checkers import BaseChecker
from pylint.checkers.utils import check_messages
from astroid import nodes


class RandomnessChecker(BaseChecker):
    """
    Detects:
      1) RNG calls (random.shuffle, np.random.rand, torch.randn, etc.) without a prior seed
      2) Calls to train_test_split or KFold without random_state or with random_state=None
    """

    name = "ml-randomness-checker"
    priority = -1
    msgs = {
        "W5503": (
            "Randomness used without setting a seed (e.g., random.seed, np.random.seed)",
            "missing-random-seed",
            "Set a random seed before using randomness to ensure reproducible results.",
        ),
        "W5504": (
            "Missing or None random_state in ML split or CV object",
            "missing-random-state",
            "Specify a non-None random_state in splitters like train_test_split, KFold, or ML estimators.",
        ),
    }

    def __init__(self, linter=None):
        super().__init__(linter)
        # Top‐level names that have been seeded (e.g. {"random", "np", "torch", "tf"})
        self.seeded_roots = set()

        # All RNG calls we’ve seen, stored as tuples of (node, full_attr_name_string)
        self.random_calls = []

        # alias → root_module, e.g. {"np": "numpy", "shuffle": "random", "rand": "numpy"}
        self.import_aliases = {}

    def visit_import(self, node):
        """
        Track:
          import random
          import numpy as np
          import torch
          import tensorflow as tf
        So that later “np.random.rand” and “tf.random.set_seed” get resolved correctly.
        """
        for real_mod, as_name in node.names:
            local_name = as_name or real_mod
            if real_mod in {"random", "numpy", "torch", "tensorflow"}:
                self.import_aliases[local_name] = real_mod

    def visit_importfrom(self, node):
        """
        Track:
          from numpy.random import rand, seed
          from random import shuffle, seed
          from torch import randn, manual_seed
          from tensorflow.random import uniform, normal, set_seed
          from sklearn.model_selection import train_test_split, KFold
        """
        mod = node.modname  # e.g. "numpy.random", "random", "sklearn.model_selection"
        if mod in {"random", "numpy.random", "torch", "tensorflow.random", "tensorflow"}:
            root = mod.split(".")[0]
            for orig_name, as_name in node.names:
                local_name = as_name or orig_name
                self.import_aliases[local_name] = root

        if mod == "sklearn.model_selection":
            for orig_name, as_name in node.names:
                local_name = as_name or orig_name
                # Mark that train_test_split/KFold came from sklearn
                self.import_aliases[local_name] = "sklearn.model_selection"

    def _full_attr_name(self, node):
        """
        Given an Attribute, recursively rebuild its dotted name.
        E.g. for “np.random.seed”, returns "np.random.seed".
        """
        if isinstance(node, nodes.Attribute):
            parent = self._full_attr_name(node.expr)
            if parent is None:
                return None
            return f"{parent}.{node.attrname}"
        elif isinstance(node, nodes.Name):
            return node.name
        else:
            return None

    @check_messages("missing-random-seed", "missing-random-state")
    def visit_call(self, node):
        """
        Called on each function call. We:
          1) Detect seed‐setting calls, e.g. random.seed(...) or np.random.seed(...)
          2) Detect RNG‐using calls, e.g. random.shuffle, np.random.rand, torch.randn, etc.
          3) Detect train_test_split(...) or KFold(...), to immediately check random_state.
        """
        func = node.func

        # (A) If it’s a dotted attribute, e.g. “np.random.seed” or “random.shuffle”
        if isinstance(func, nodes.Attribute):
            full_name = self._full_attr_name(func)
            if not full_name:
                return

            # (A1) seed‐setters:
            seed_functions = {
                "random.seed",
                "numpy.random.seed",
                "np.random.seed",
                "torch.manual_seed",
                "tensorflow.random.set_seed",
                "tf.random.set_seed",
                "tensorflow.set_random_seed",
            }
            if full_name in seed_functions:
                root = full_name.split(".")[0]  # “random” or “np” or “torch” or “tf”
                self.seeded_roots.add(root)
                return

            # (A2) RNG‐users:
            rng_functions = {
                "random.random",
                "random.shuffle",
                "random.choice",
                "numpy.random.rand",
                "numpy.random.randn",
                "np.random.rand",
                "np.random.randn",
                "np.random.shuffle",
                "np.random.uniform",
                "np.random.normal",
                "torch.rand",
                "torch.randn",
                "torch.randint",
                "torch.randperm",
                "tensorflow.random.uniform",
                "tensorflow.random.normal",
                "tf.random.uniform",
                "tf.random.normal",
                "tensorflow.random.shuffle",
                "tf.random.shuffle",
            }
            if full_name in rng_functions:
                # Stash for “late” warning in close()
                self.random_calls.append((node, full_name))
                return

            # (A3) ML‐splitters invoked as attributes, e.g. “sklearn.model_selection.train_test_split”
            if func.attrname in {"train_test_split", "KFold"}:
                self._check_random_state_arg(node)
                return

        # (B) If it’s a plain Name call, e.g. “shuffle(...)” or “rand(...)” or “train_test_split(...)”
        if isinstance(func, nodes.Name):
            name = func.name

            # (B1) train_test_split / KFold imported from sklearn:
            if name in {"train_test_split", "KFold"} and self.import_aliases.get(name) == "sklearn.model_selection":
                self._check_random_state_arg(node)
                return

            # (B2) “from numpy.random import rand” → name = “rand”, import_aliases[“rand”] == “numpy”
            root = self.import_aliases.get(name)
            if root == "numpy" and name in {"rand", "randn", "shuffle", "uniform", "normal"}:
                full_name = f"{root}.random.{name}"
                self.random_calls.append((node, full_name))
                return

            # (B3) “from random import shuffle” → name = “shuffle”, import_aliases[“shuffle”] == “random”
            if root == "random" and name in {"shuffle", "random", "choice"}:
                full_name = f"{root}.{name}"
                self.random_calls.append((node, full_name))
                return

            # (B4) “from torch import rand, randn” → name = “rand” or “randn”; import_aliases[name] == “torch”
            if root == "torch" and name in {"rand", "randn", "randint", "randperm"}:
                full_name = f"{root}.{name}"
                self.random_calls.append((node, full_name))
                return

            # (B5) “from tensorflow.random import uniform, normal, shuffle” → name = “uniform”, etc; import_aliases[name] == “tensorflow” or “tensorflow.random”
            if root in {"tensorflow", "tensorflow.random"} and name in {"uniform", "normal", "shuffle"}:
                full_name = f"tensorflow.random.{name}"
                self.random_calls.append((node, full_name))
                return

        # Otherwise: not relevant to randomness detection

    @check_messages("missing-random-state")
    def _check_random_state_arg(self, node):
        """
        Called from inside visit_call when we detect train_test_split or KFold.
        Warn if no “random_state” keyword, or if random_state=None.
        """
        keywords = {kw.arg: kw.value for kw in (node.keywords or [])}
        if "random_state" not in keywords:
            self.add_message("missing-random-state", node=node)
        else:
            val = keywords["random_state"]
            if isinstance(val, nodes.Const) and val.value is None:
                self.add_message("missing-random-state", node=node)

    @check_messages("missing-random-seed")
    def close(self):
        """
        After the entire module is processed, go over every RNG call we stashed.
        If its “root” was never seeded, emit W5503 at that call’s node.
        """
        for call_node, full_name in self.random_calls:
            root = full_name.split(".")[0]  # e.g. “np” or “random” or “torch” or “tf”
            if root not in self.seeded_roots:
                self.add_message("missing-random-seed", node=call_node)


def register(linter):
    """
    Hook so that Pylint will register this checker.
    """
    print(">>> ml_pylint REGISTERED <<<")
    linter.register_checker(RandomnessChecker(linter))
