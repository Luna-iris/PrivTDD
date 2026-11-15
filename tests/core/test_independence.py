"""Installable core must not import the original research workspace."""
import ast
import importlib
import subprocess
import sys
import unittest
from pathlib import Path

from privtdd.io.identity import source_files, source_digest


class IndependenceTests(unittest.TestCase):
    def test_all_core_modules_import_without_research_package(self):
        root = Path(importlib.import_module('privtdd.config').__file__).parent
        for file in source_files():
            module = 'privtdd.' + '.'.join(file.relative_to(root).with_suffix('').parts)
            importlib.import_module(module)
        self.assertFalse(any(name == 'reconstruct_overall' or name.startswith('reconstruct_overall.')
                             for name in sys.modules))

    def test_no_legacy_import_or_path_mutation_in_core(self):
        for file in source_files():
            tree = ast.parse(file.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                names = ([x.name for x in node.names] if isinstance(node, ast.Import)
                         else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                self.assertFalse(any(name.startswith(('reconstruct_overall', 'c9_', 'DataProcess')) for name in names), file)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertFalse(ast.unparse(node.func).startswith('sys.path.'), file)

    def test_source_identity_covers_semantic_modules(self):
        names = {p.parent.name for p in source_files()}
        self.assertTrue({'degree', 'references', 'correction', 'posterior', 'release', 'reporting',
                         'base_distribution', 'distributions', 'numerics', 'io'} <= names)
        self.assertEqual(len(source_digest()), 64)
        self.assertEqual(source_digest(), source_digest())


if __name__ == '__main__':
    unittest.main()
