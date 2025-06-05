"""
Pylint checker to ensure that random seeds are set when using random-like modules
(e.g., random, numpy, torch, tensorflow) to enforce reproducibility.
"""

from pylint.checkers import BaseChecker


class MissingRandomSeedChecker(BaseChecker):
    """
    A checker that verifies if modules introducing randomness are imported
    without calling their respective seed-setting functions.
    """

    name = "missing-random-seed-checker"
    priority = -1
    msgs = {
        "W9002": (
            "No random seed set for module(s): %s",
            "missing-random-seed",
            "Random seed not set; results may be nondeterministic.",
        ),
    }

    ALIAS_MAP = {"np": "numpy", "tf": "tensorflow"}

    def __init__(self, linter=None):
        super().__init__(linter)
        self.module_node = None
        self.imported_modules = set()
        self.seed_set_modules = set()

    def visit_module(self, node):
        """
        Reset state at the start of each module.
        """
        self.imported_modules.clear()
        self.seed_set_modules.clear()
        self.module_node = node

    def visit_import(self, node):
        """
        Track standard import statements.
        """
        for name, alias in node.names:
            self.imported_modules.add(alias or name)

    def visit_importfrom(self, node):
        """
        Track 'from module import ...' statements.
        """
        self.imported_modules.add(node.modname)

    def visit_call(self, node):
        """
        Detect known seed-setting function calls.
        """
        try:
            func_name = node.func.as_string()
        except AttributeError:
            return

        if func_name in {
            "random.seed",
            "numpy.random.seed",
            "np.random.seed",
            "torch.manual_seed",
            "tensorflow.random.set_seed",
            "tf.random.set_seed",
        }:
            self.seed_set_modules.add(func_name.split(".")[0])

    def _normalize(self, modules):
        """
        Normalize aliases (e.g., np → numpy).
        """
        return {self.ALIAS_MAP.get(m, m) for m in modules}

    def close(self):
        """
        At end of module, emit a warning if any imported random-like module is missing a seed.
        """
        required = {"random", "numpy", "torch", "tensorflow"}
        imported = self._normalize(self.imported_modules)
        seeded = self._normalize(self.seed_set_modules)
        if (missing := (imported & required) - seeded):
            self.add_message(
                "missing-random-seed",
                node=self.module_node,
                args=(", ".join(sorted(missing)),),
            )


def register(linter):
    """
    Register the custom checker with Pylint.
    """
    linter.register_checker(MissingRandomSeedChecker(linter))
