from robocasa.utils.dataset_registry import (
    get_ds_path,
    SINGLE_STAGE_TASK_DATASETS,
    MULTI_STAGE_TASK_DATASETS,
)
from robocasa.scripts.playback_dataset import get_env_metadata_from_dataset
from robosuite import load_controller_config
import os
import robosuite
import imageio
import numpy as np
from tqdm import tqdm
from termcolor import colored
from scipy.spatial.transform import Rotation as R

def scripted_policy(obs):
    # Extract relevant information from observations
    eef_pos = np.array(obs['robot0_eef_pos'])         # End-effector position
    eef_quat = np.array(obs['robot0_eef_quat'])       # End-effector orientation (quaternion)
    
    obj_pos = np.array(obs['drawer_obj_pos'])         # Object position (e.g., drawer)
    obj_quat = np.array(obs['drawer_obj_quat'])       # Object orientation (quaternion)
    
    # Compute position difference (object to end-effector)
    position_diff = obj_pos - eef_pos
    distance = np.linalg.norm(position_diff)

    # Compute rotation difference (object orientation to end-effector orientation)
    eef_rot = R.from_quat(eef_quat)
    obj_rot = R.from_quat(obj_quat)
    rot_diff = obj_rot * eef_rot.inv()
    rot_diff_axis_angle = rot_diff.as_rotvec()

    # Set control gains (adjust these based on your system's tuning)
    threshold = 0.1  # Distance threshold to stop moving towards the object
    Kp_pos = 1.0  # Position proportional gain
    Kp_rot = 0.5  # Rotation proportional gain

    # Compute the control actions
    action = np.zeros(12)

    # Position control (P control, no velocity)
    if distance > threshold:
        action[:3] = Kp_pos * position_diff  # Proportional control for position
    else:
        return action # Stop moving if close enough

    # Orientation control (P control)
    action[3:6] = Kp_rot * rot_diff_axis_angle  # Control for orientation

    # Stiffness values (for stability)
    action[6:9] = [10, 10, 10]  # Position stiffness
    action[9:12] = [5, 5, 5]  # Orientation stiffness
    
    return action




def create_env(
    env_name,
    # robosuite-related configs
    robots="PandaMobile",
    controllers="OSC_POSE",
    camera_names=[
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ],
    camera_widths=128,
    camera_heights=128,
    seed=None,
    # robocasa-related configs
    obj_instance_split=None,
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=None,
    layout_ids=None,
    style_ids=None,
):
    controller_configs = load_controller_config(default_controller=controllers)

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_configs,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=False,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        layout_ids=layout_ids,
        style_ids=style_ids,
        translucent_robot=False,
    )

    env = robosuite.make(**env_kwargs)
    return env


def run_random_rollouts(env, num_rollouts, num_steps, video_path=None):
    video_writer = None
    if video_path is not None:
        video_writer = imageio.get_writer(video_path, fps=20)

    info = {}
    num_success_rollouts = 0
    for rollout_i in tqdm(range(num_rollouts)):
        obs = env.reset()
        for step_i in range(num_steps):
            # sample and execute random action
            # action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
            action = scripted_policy(obs)
            # print('1111111111111111111')
            print(action)
            obs, _, _, _ = env.step(action)
            print('######################################')
            # print(type(obs))
            print(obs)

            if video_writer is not None:
                video_img = env.sim.render(
                    height=512, width=512, camera_name="robot0_agentview_center"
                )[::-1]
                video_writer.append_data(video_img)

            if env._check_success():
                num_success_rollouts += 1
                break

    if video_writer is not None:
        video_writer.close()
        print(colored(f"Saved video of rollouts to {video_path}", color="yellow"))

    info["num_success_rollouts"] = num_success_rollouts

    return info


if __name__ == "__main__":
    # select random task to run rollouts for
    env_name = np.random.choice(
        list(SINGLE_STAGE_TASK_DATASETS) + list(MULTI_STAGE_TASK_DATASETS)
    )
    env = create_eval_env(env_name=env_name)
    info = run_random_rollouts(
        env, num_rollouts=3, num_steps=100, video_path="/tmp/test.mp4"
    )
