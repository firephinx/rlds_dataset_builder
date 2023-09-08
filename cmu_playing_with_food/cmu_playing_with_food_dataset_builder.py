from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from pathlib import Path


class CmuPlayingWithFood(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'finger_vision_1': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Finger Vision 1 RGB observation.',
                        ),
                        'finger_vision_2': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Finger Vision 2 RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Robot state, consists of [6x end-effector force].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Robot action, consists of [3x end-effector position, '
                            '4x end-effector quaternion, 1x gripper open/close].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(),
        }

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        from cmu_playing_with_food.playing_with_food_helpers import load_kevin_data
    
        def _parse_example(episode_path_instruction: Tuple[str, str, int]):
            episode_path, skill_type = Path(episode_path_instruction[0]), episode_path_instruction[1]
            episode_idx = f'{episode_path_instruction[2]}'

            episode = []
            steps = 0
            for step, step_data in load_kevin_data(episode_path, skill_type, load_images=True):
                if step == 0:
                    language_embedding = self._embed([step_data['language_instruction']])[0].numpy()
                step_data['language_embedding'] = np.copy(language_embedding)
                episode.append(step_data)
                steps += 1

            # create output data sample
            relative_path_idx = episode_path.parts.index('playing_with_food')
            episode_str = '/'.join(episode_path.parts[relative_path_idx + 1:])
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_str
                }
            }
            return episode_idx, sample


        playing_with_food_path = Path('/home/klz/Documents/playing_with_food/playing_data/')

        step_count, episode_count = 0, 0

        skill_types = ['grasp', 'release', 'push_down']

        for food_name in playing_with_food_path.iterdir():
            for slice_num in food_name.iterdir():
                if slice_num.is_file():
                    continue
                for trial_num in slice_num.iterdir():
                    for skill_type in skill_types:
                        yield _parse_example((trial_num, skill_type, episode_count))
                        episode_count += 1
                        
                print(f"Total episodes: {episode_count} steps: {step_count}")

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

