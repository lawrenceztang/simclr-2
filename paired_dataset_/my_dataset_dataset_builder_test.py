"""paired_dataset_ dataset."""

import tensorflow_datasets as tfds
from . import paired_dataset_builder


class MyDatasetTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for paired_dataset_ dataset."""
  # TODO(paired_dataset_):
  DATASET_CLASS = paired_dataset_builder.Builder
  SPLITS = {
      'all': 4
  }
  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
