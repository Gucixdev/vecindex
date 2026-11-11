"""Tests of the Python package against a real store on disk: python -m unittest discover -s tests (from bindings/python,
with the package built: pip install . or python setup.py build_ext --inplace)."""
import array
import math
import os
import shutil
import tempfile
import unittest

import vecindex

DIM = 8


def vec(i):
    """A vector that is a function of i: distinct directions, so a row's own vector finds it first."""
    return [math.sin(i * 1.3 + d * 0.7) + (0.5 if d == i % DIM else 0.0) for d in range(DIM)]


class Store(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="vecindex_py_")
        self.path = os.path.join(self.dir, "t.vxi")

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_version(self):
        self.assertEqual(vecindex.version(), vecindex.__version__)

    def test_add_search_reopen(self):
        with vecindex.open(self.path, dim=DIM) as db:
            for i in range(1, 201):
                db.add(i, vec(i), text="chunk %d about %s" % (i, "cats" if i % 2 else "ships") if i % 5 == 0 else None)
            self.assertEqual(len(db), 200)
            hits = db.search(vec(42), k=3)
            self.assertEqual(hits[0][0], 42)
            self.assertGreater(hits[0][1], 0.99)
            self.assertEqual(db.text(15), "chunk 15 about cats")
            self.assertIsNone(db.text(16))
            db.sync()
        with vecindex.open(self.path) as db:                     # dim 0: the file says
            self.assertEqual(db.dim, DIM)
            self.assertEqual(len(db), 200)
            self.assertEqual(db.search(vec(7), k=1)[0][0], 7)
            self.assertEqual(db.text(20), "chunk 20 about ships")
            got = db.get(9)
            n = math.sqrt(sum(x * x for x in vec(9)))
            self.assertTrue(all(abs(a - b / n) < 1e-5 or abs(a - b) < 1e-5 for a, b in zip(got, vec(9))))

    def test_buffers_and_batch(self):
        with vecindex.open(None, dim=DIM) as db:
            flat = array.array("f")
            for i in range(10):
                flat.extend(vec(i))
            db.add_many(list(range(10)), flat)                   # one flat float32 buffer
            db.add_many([10, 11], [vec(10), vec(11)])            # a list of lists
            db.add(12, array.array("f", vec(12)))
            self.assertEqual(len(db), 13)
            self.assertEqual(db.search(array.array("f", vec(11)), k=1)[0][0], 11)
            try:
                import numpy as np
            except ImportError:
                np = None
            if np is not None:
                db.add(13, np.asarray(vec(13), dtype=np.float32))
                db.add(14, np.asarray(vec(14), dtype=np.float64))
                self.assertEqual(db.search(np.asarray(vec(14), dtype=np.float32), k=1)[0][0], 14)

    def test_errors(self):
        with vecindex.open(None, dim=DIM) as db:
            with self.assertRaises(ValueError):
                db.add(1, [1.0, 2.0])
            with self.assertRaises(vecindex.VecindexError):
                db.add(1, [float("nan")] * DIM)
            self.assertIsNone(db.get(99))
            self.assertFalse(db.delete(99))
            with self.assertRaises(ValueError):
                db.search(vec(1), text="x", ns=1)
        db.close()
        with self.assertRaises(vecindex.VecindexError):
            len(db)
        with open(self.path, "wb") as f:
            f.write(b"not a store at all, but some bytes")
        with self.assertRaises(vecindex.VecindexError):
            vecindex.open(self.path)

    def test_conditions(self):
        with vecindex.open(self.path, dim=DIM) as db:
            for i in range(1, 301):
                db.add(i, vec(i), text=("the rare word zyxwv" if i == 77 else "plain text %d" % i))
                db.set_num(i, i * 10)
                db.set_ns(i, "even" if i % 2 == 0 else "odd")
                if i % 3 == 0:
                    db.add_labels(i, "dog" if i % 6 == 0 else "cat")
            db.label("dog", "pet")
            db.label("cat", "pet")
            q = vec(150)
            self.assertTrue(all(i % 2 == 0 for i, _ in db.search(q, k=20, ns="even")))
            self.assertEqual(db.count_ns("odd"), 150)
            both = db.search(q, k=20, ns=["odd", "even"])
            self.assertEqual(both[0][0], 150)
            self.assertTrue(all(1000 <= i * 10 <= 1500 for i, _ in db.search(q, k=20, num=(1000, 1500))))
            self.assertEqual(db.num(5), 50)
            self.assertEqual(db.ns(5), vecindex.key("odd"))
            pets = db.search(q, k=50, labels={"all": "pet"})
            self.assertTrue(pets and all(i % 3 == 0 for i, _ in pets))
            self.assertTrue(all(i % 6 == 0 for i, _ in db.search(q, k=50, labels={"all": ["dog"]})))
            self.assertEqual(db.search(None, k=1, text="zyxwv")[0][0], 77)
            db.update(150, vec(151))
            self.assertEqual(db.num(150), 1500)
            self.assertEqual(db.delete_ns("odd"), 150)
            self.assertEqual(len(db), 150)
            db.save()
        with vecindex.open(self.path) as db:
            self.assertEqual(len(db), 150)
            self.assertEqual(db.num(150), 1500)

    def test_grouped(self):
        with vecindex.open(None, dim=DIM) as db:
            for doc in range(1, 6):
                for chunk in range(4):
                    db.add(doc << 20 | chunk, vec(doc * 10 + chunk))
            hits = db.search(vec(30), k=3, group=20)
            docs = [i >> 20 for i, _ in hits]
            self.assertEqual(len(set(docs)), len(docs))
            self.assertEqual(docs[0], 3)


if __name__ == "__main__":
    unittest.main()
