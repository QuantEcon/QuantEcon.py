"""
Tests for game_theory/game_converters.py

"""
import os
from tempfile import NamedTemporaryFile
from numpy.testing import assert_string_equal
from quantecon.game_theory import NormalFormGame, GAMWriter, to_gam


class TestGAMWriter:
    def setup_method(self):
        nums_actions = (2, 2, 2)
        g = NormalFormGame(nums_actions)
        g[0, 0, 0] = (0, 8, 16)
        g[1, 0, 0] = (1, 9, 17)
        g[0, 1, 0] = (2, 10, 18)
        g[1, 1, 0] = (3, 11, 19)
        g[0, 0, 1] = (4, 12, 20)
        g[1, 0, 1] = (5, 13, 21)
        g[0, 1, 1] = (6, 14, 22)
        g[1, 1, 1] = (7, 15, 23)
        self.g = g

        self.s_desired = """\
3
2 2 2

0. 1. 2. 3. 4. 5. 6. 7. \
8. 9. 10. 11. 12. 13. 14. 15. \
16. 17. 18. 19. 20. 21. 22. 23."""

    def test_to_file(self):
        with NamedTemporaryFile(delete=False) as tmp_file:
            temp_path = tmp_file.name
            GAMWriter.to_file(self.g, temp_path)

        with open(temp_path, 'r') as f:
            s_actual = f.read()
        assert_string_equal(s_actual, self.s_desired + '\n')

        os.remove(temp_path)

    def test_to_string(self):
        s_actual = GAMWriter.to_string(self.g)

        assert_string_equal(s_actual, self.s_desired)

    def test_to_gam(self):
        s_actual = to_gam(self.g)
        assert_string_equal(s_actual, self.s_desired)

        with NamedTemporaryFile(delete=False) as tmp_file:
            temp_path = tmp_file.name
            to_gam(self.g, temp_path)

        with open(temp_path, 'r') as f:
            s_actual = f.read()
        assert_string_equal(s_actual, self.s_desired + '\n')

        os.remove(temp_path)
