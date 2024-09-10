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

def scripted_policy(obs, has_grasped=False, moved_to_grasp_pos=False, lifted=False):
    # Extract relevant information from observations
    eef_pos = np.array(obs['robot0_eef_pos'])         # End-effector position
    eef_quat = np.array(obs['robot0_eef_quat'])       # End-effector orientation (quaternion)
    
    obj_pos = np.array(obs['cookware_pos'])         # Object position (e.g., drawer)
    obj_quat = np.array(obs['cookware_quat'])       # Object orientation (quaternion)

    # Drawer: drawer_obj
    # SinkFaucet: distr_counter_0
    # Coffee: obj
    # TurnOnStove: cookware
    # Door: door_obj
    # CoffeeServeMug: obj
    # Microwave: obj
    
    # Compute position difference (object to end-effector)
    position_diff = obj_pos - eef_pos
    distance = np.linalg.norm(position_diff)

    # Compute rotation difference (object orientation to end-effector orientation)
    # rot_diff = robosuite.utils.transform_utils.quat_multiply(obj_quat, robosuite.utils.transform_utils.quat_inverse(eef_quat))
    # rot_diff_axis_angle = robosuite.utils.transform_utils.quat2axisangle(rot_diff)
    eef_rot = R.from_quat(eef_quat)
    obj_rot = R.from_quat(obj_quat)
    rot_diff = obj_rot * eef_rot.inv()
    rot_diff_axis_angle = rot_diff.as_rotvec()

    # Set control gains (adjust these based on your system's tuning)
    threshold = 0.02  # Distance threshold to stop moving towards the object
    Kp_pos = 1  # Position proportional gain
    Kp_rot = 0.0001  # Rotation proportional gain

    # Compute the control actions
    action = np.zeros(12)

    lift_distance = 1
    original_z = 0

    # Phase 1: Move toward the object center
    if not has_grasped:
        if not moved_to_grasp_pos:
            # Move towards the object center first
            if distance > threshold:
                action[:3] = Kp_pos * position_diff  # Proportional control for position
            else:
                # When within the threshold, move 10 cm sideways (x-axis) to prepare for grasp
                moved_to_grasp_pos = True
                original_z = action[2]
        else:
            # Move 10 cm sideways along the x-axis (or y-axis if desired)
            sideways_distance = 0.15  # 10 cm movement sideways along the x-axis
            sideways_pos = np.array([sideways_distance, 0, 0])  # Moving 10 cm in x-direction
            move_diff = sideways_pos  # The sideways movement is predefined, relative to the current position

            action[:3] = Kp_pos * move_diff  # Apply the sideways action

            # Once the sideways movement is done, close the gripper to grasp the object
            if np.linalg.norm(move_diff) < threshold:
                action[6:7] = [1]  # Close the gripper to grasp
                has_grasped = True  # Mark that the object has been grasped

    # Phase 2: Lift the object after grasping
    elif has_grasped and not lifted:
        # Lift the object by moving upwards (z-axis only)
        lift_target_z = original_z + lift_distance  # Increase only the z-coordinate for lifting

        lift_diff_z = lift_target_z - eef_pos[2]

        # Apply the lifting action only on the z-axis (3rd component of action)
        action[2] = 0.0001 * lift_diff_z

        # Keep the gripper closed while lifting
        action[6:7] = [1]  # Keep gripper closed

        # Check if the robot has lifted the object enough
        if abs(lift_diff_z) < threshold:
            lifted = True  # Mark that the object has been lifted


    # Orientation control (align with object)
    action[3:6] = Kp_rot * rot_diff_axis_angle  # Control for orientation

    # Other controls (base, torso, wheel)
    action[7:10] = [0, 0, 0]  # base
    action[10:11] = 0         # torso
    action[11:12] = 0         # wheel (optional)

    return action, has_grasped, moved_to_grasp_pos, lifted




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
    has_grasped = False
    move_to_grasp_pose = False
    lifted = False
    for rollout_i in tqdm(range(num_rollouts)):
        obs = env.reset()
        for step_i in range(num_steps):
            # sample and execute random action
            # action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
            action, has_grasped, move_to_grasp_pose, lifted = scripted_policy(obs, has_grasped, move_to_grasp_pose, lifted)
            if np.allclose(action[:3], [0, 0, 0], atol=1e-4): 
                print("REACHED")
            else:
                print(action)
                print('######################################')
                print(obs)
            # print('1111111111111111111')
            # print(action)
            obs, _, _, _ = env.step(action)
            # print('######################################')
            # print(type(obs))
            # print(obs)

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
