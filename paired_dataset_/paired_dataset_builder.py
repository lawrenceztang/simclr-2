"""paired_dataset_ dataset."""

import tensorflow_datasets as tfds
import os
from collections import defaultdict

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for paired_dataset_ dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = "Download at https://github.com/cvdfoundation/kinetics-dataset"

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(paired_dataset_): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image0': tfds.features.Image(shape=(None, None, 3)),
            'image1': tfds.features.Image(shape=(None, None, 3)),
            "label": tfds.features.ClassLabel(num_classes=10),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys= ("image0", "image1"),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    # TODO(paired_dataset_): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    path = dl_manager.manual_dir
    # TODO(paired_dataset_): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path),
        'test': self._generate_examples(path),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(paired_dataset_): Yields (key, example) tuples from the dataset

    files_dict = defaultdict(list)

    for file_name in os.listdir(path):
        if file_name.endswith('0.jpg') or file_name.endswith('1.jpg'):
            # Remove the last character (postfix) to group files
            base_name = file_name[:-5]
            files_dict[base_name].append(file_name)

    # Create pairs
    id = 0
    for base_name, files in files_dict.items():
        if len(files) == 2:
            # Ensure there's a pair of '0' and '1'
            if files[0][-5] != files[1][-5]:
                yield id, {
                    'image0': os.path.join(path, files[0]),
                    'image1': os.path.join(path, files[1]),
                    'label': 0
                }
            id += 1
