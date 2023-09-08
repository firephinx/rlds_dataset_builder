import numpy as np
import os
from pathlib import Path

import pickle
from PIL import Image
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
import autolab_core

import cv2
import os

def load_all_frames(video_path, debug: bool = False):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return []

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    frames = []

    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            if debug:
                cv2.imshow('video',frame)
                cv2.waitKey(0)
            #cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return frames

def convert_quaternion_to_euler(quat: np.ndarray):
    # Since we use pyquaternion, we need to convert to scipy rotation
    r = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]])
    return r.as_euler('zyx')


def load_image(image_path: Path, resize_ratio: float = 1.0):
    img = Image.open(str(image_path))
    if resize_ratio < 1.0:
        img = img.resize([int(resize_ratio * s) for s in img.size], Image.LANCZOS)

    return np.asarray(img)[:, :, :3]


def load_kevin_data(demo_path: Path, skill_type: str, load_images: bool = False):
    info_pickle_path = demo_path / 'trial_info.npz'
    realsense_img_path = demo_path / f'{skill_type}_realsense_rgb.avi'
    finger_vision_1_img_path = demo_path / f'{skill_type}_finger_vision_1.avi'
    finger_vision_2_img_path = demo_path / f'{skill_type}_finger_vision_2.avi'

    #demo_path_str = str(demo_path)
    target_index = demo_path.parts.index('playing_data') + 1
    food_name = demo_path.parts[target_index]

    realsense_frames = load_all_frames(realsense_img_path)
    if skill_type == 'push_down':
        finger_vision_1_frames = [np.zeros_like(realsense_frames[0]) for _ in range(len(realsense_frames))]
        finger_vision_2_frames = [np.zeros_like(realsense_frames[0]) for _ in range(len(realsense_frames))]
    else:
        finger_vision_1_frames = load_all_frames(finger_vision_1_img_path)
        finger_vision_2_frames = load_all_frames(finger_vision_2_img_path)

    min_length = min([len(x) for x in [realsense_frames,finger_vision_1_frames,finger_vision_2_frames]])

    if min_length == 0:
        return

    proprio = np.load(info_pickle_path, allow_pickle=True)

    positions = np.zeros((min_length,3))
    orientations = np.zeros((min_length,4))
    orientations[:,0] = 1
    if skill_type == 'grasp':
        lang_instruction = 'Grasp the ' + food_name + ' slice.'

        for i in range(3):
            positions[:, i] = proprio['object_world_position'].tolist()[i]
        forces = np.zeros((min_length,6))
        grasps = np.zeros((min_length,1))
    elif skill_type == 'release':
        lang_instruction = 'Release the ' + food_name + ' slice.'
        for i in range(3):
            positions[:, i] = proprio['intermediate_robot_position'].tolist()[i]
        forces = np.zeros((min_length,6))
        grasps = np.ones((min_length,1))
    elif skill_type == 'push_down':
        lang_instruction = 'Press down on the ' + food_name + ' slice.'
        all_positions = proprio['push_down_robot_positions']
        all_forces = proprio['push_down_robot_forces']

        positions = all_positions[np.floor(np.linspace(0, all_positions.shape[0]-1, min_length)).astype(int)]
        forces = all_forces[np.floor(np.linspace(0, all_forces.shape[0]-1, min_length)).astype(int)]
        grasps = np.zeros((min_length,1))
    else:
        raise ValueError
    
    actions = np.c_[positions, orientations, grasps]
    state = np.copy(forces)

    
    episode_len = min_length
    for t in range(episode_len - 1):
        data_t = {
            'observation': {
                'image': realsense_frames[t],
                'finger_vision_1': finger_vision_1_frames[t],
                'finger_vision_2': finger_vision_2_frames[t],
                'state': state[t].astype(np.float32),
            },
            'action': actions[t].astype(np.float32),
            'discount': 1.0,
            'reward': float(t == (episode_len - 1)),
            'is_first': t == 0,
            'is_last': t == (episode_len - 1),
            'is_terminal': t == (episode_len - 1),
            'language_instruction': lang_instruction 
            # 'language_embedding': language_embedding,
        }
        yield t, data_t



def main():
    playing_with_food_path = Path('/home/klz/Documents/playing_with_food/playing_data/')

    step_count, episode_count = 0, 0

    skill_types = ['grasp', 'release', 'push_down']
    for food_name in playing_with_food_path.iterdir():
        for slice_num in food_name.iterdir():
            if slice_num.is_file():
                continue
            for trial_num in slice_num.iterdir():
                for skill_type in skill_types:
                    #load_kevin_data(trial_num, skill_type, load_images=False)
                    for step_data in load_kevin_data(trial_num, skill_type, load_images=False):
                        step_count += 1
                    episode_count += 1
                    
            print(f"Total episodes: {episode_count} steps: {step_count}")


if __name__ == '__main__':
    main()
