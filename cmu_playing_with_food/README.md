## CMU Playing with Food Dataset

The dataset contains 4200 episodes (~400K steps) for three different skills Grasp, Press Down, and Release on food slices. There are 28 types of food slices where some classes are repeated but were boiled. In each food type, there are 10 slices with 5 trials of each skill on each slice.

There are multiple camera views, one statically mounted third-person view and 2 finger vision views that display how the food item is crushed.
All image data was recorded at 15Hz. The proprioceptive data was recorded at around 50Hz, but is subsampled for this dataset to be 15Hz and timestep aligned with the image data.

The press down skill does not have any finger vision data so there are only images of zeros. Meanwhile the Grasp and Push Down skills have zero forces because they were not recorded and only contain views of the robot closing and opening the gripper.

More information of the dataset can be found in this website: [Playing with Food](https://sites.google.com/view/playing-with-food/).

Warning: Some of the language commands may have underscores. I will attempt to remove them in a future update. In addition, we have audio data, but I will add them later.

**Instruction**: Grasp the apple slice.

![Grasp apple slice third-person](./static/grasp_apple.gif) ![Grasp apple slice finger vision 1](./static/grasp_apple_1.gif) ![Grasp apple slice finger vision 2](./static/grasp_apple_2.gif)


**Instruction**: Press down on the apple slice

![Press down apple slice third-person](./static/press_down_apple.gif) 

**Instruction**: Release the apple slice

![Release apple slice third-person](./static/release_apple.gif) ![Release apple slice finger vision 1](./static/release_apple_1.gif) ![Release apple slice finger vision 2](./static/release_apple_2.gif)