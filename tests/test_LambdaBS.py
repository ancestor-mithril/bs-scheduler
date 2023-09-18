import unittest

from torch.optim.lr_scheduler import LambdaLR


class TestLambdaBS(unittest.TestCase):
    def setUp(self):
        pass

    def test_true(self):
        x = LambdaLR()
        x = x.step()
        self.assertTrue(True)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()
