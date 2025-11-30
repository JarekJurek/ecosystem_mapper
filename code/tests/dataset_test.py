import unittest
import pandas as pd
import torch

from code.dataset.dataset import EcosystemDataset, get_dataloaders, default_collate
from code.dataset.sweco_group_of_variables import sweco_variables_dict

CSV_PATH = "data/dataset_split.csv"

class TestEcosystemDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(CSV_PATH)
        # meta columns as defined in dataset.py
        cls.meta_cols = {"id", "x", "y", "split", "EUNIS_cls", "EUNIS_label"}
        cls.all_var_cols = [c for c in cls.df.columns if c not in cls.meta_cols]

    def test_split_lengths(self):
        for split in ["train", "val", "test"]:
            expected_len = (self.df[self.df["split"] == split]).shape[0]
            ds = EcosystemDataset.from_split(
                subset=split,
                csv_path=CSV_PATH,
                image_dir=None,  # images disabled
                load_images=False,
                variable_selection="all",
                return_label=True,
            )
            self.assertEqual(len(ds), expected_len, f"Length mismatch for split {split}")
            # Ensure all rows in dataset have only the requested split
            self.assertTrue(all(ds.df["split"] == split))

    def test_variable_selection_all(self):
        ds = EcosystemDataset(
            csv_path=CSV_PATH,
            image_dir=None,
            load_images=False,
            variable_selection="all",
            return_label=True,
        )
        self.assertEqual(set(ds.var_cols), set(self.all_var_cols))
        sample = ds[0]
        self.assertIn("variables", sample)
        self.assertEqual(sample["variables"].shape[0], len(self.all_var_cols))

    def test_variable_selection_group(self):
        # choose a known group key, e.g. 'geol'
        group = "geol"
        expected = [c for c in sweco_variables_dict[group] if c in self.df.columns]
        ds = EcosystemDataset(
            csv_path=CSV_PATH,
            image_dir=None,
            load_images=False,
            variable_selection=group,
            return_label=False,
        )
        self.assertEqual(ds.var_cols, expected)
        sample = ds[0]
        self.assertIn("variables", sample)
        self.assertEqual(sample["variables"].shape[0], len(expected))
        # All variable names should appear in dataset columns
        for v in expected:
            self.assertIn(v, self.df.columns)

    def test_variable_selection_list_and_dedup(self):
        # mix group name and explicit column, plus duplicate
        group = "hydro"
        extra_col = self.all_var_cols[0]
        ds = EcosystemDataset(
            csv_path=CSV_PATH,
            image_dir=None,
            load_images=False,
            variable_selection=[group, extra_col, group, extra_col],
            return_label=False,
        )
        # expected = hydro group vars + extra_col, deduplicated
        expected_set = set([c for c in sweco_variables_dict[group] if c in self.df.columns] + [extra_col])
        self.assertEqual(set(ds.var_cols), expected_set)

    def test_no_variables(self):
        ds = EcosystemDataset(
            csv_path=CSV_PATH,
            image_dir=None,
            load_images=False,
            variable_selection=None,
            return_label=True,
        )
        sample = ds[0]
        self.assertNotIn("variables", sample)
        self.assertIn("label", sample)

    def test_dataloaders_basic(self):
        loaders = get_dataloaders(
            csv_path=CSV_PATH,
            image_dir=None,
            load_images=False,
            variable_selection="all",
            batch_size=4,
            num_workers=0,
        )
        self.assertEqual(set(loaders.keys()), {"train", "val", "test"})
        batch = next(iter(loaders["train"]))
        # images disabled
        self.assertIsNone(batch["images"])
        self.assertIsInstance(batch["variables"], torch.Tensor)
        self.assertEqual(batch["variables"].shape[0], 4)
        # Check labels present
        self.assertIsInstance(batch["labels"], torch.Tensor)

    def test_collate_padding(self):
        # Create artificial batch with differing variable lengths by selecting different groups
        ds_geol = EcosystemDataset(CSV_PATH, image_dir=None, load_images=False, variable_selection="geol", return_label=False)
        ds_hydro = EcosystemDataset(CSV_PATH, image_dir=None, load_images=False, variable_selection="hydro", return_label=False)
        item1 = ds_geol[0]
        item2 = ds_hydro[1]
        batch = default_collate([item1, item2])
        vars_t = batch["variables"]
        self.assertIsInstance(vars_t, torch.Tensor)
        # Should have shape (2, max_len)
        self.assertEqual(vars_t.shape[0], 2)
        max_len = max(len(sweco_variables_dict["geol"]), len(sweco_variables_dict["hydro"]))
        # Hydro may have fewer columns; padded with zeros
        self.assertEqual(vars_t.shape[1], max_len)

if __name__ == "__main__":
    unittest.main()
