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
from robocasa.utils.primatives import pick, place, reach, pour, twist, open, close, defrost, placeinoven
from collections import deque

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
    randomize_cameras=True,
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
    org_x = 0
    org_y = 0
    org_z = 0
    moved_down = False
    placed = False
    poured = False
    moved_forward = False
    moved_back = False
    pushed_forward = False
    released_grasp = False
    door_opened = False
    door_closed = False
    obj_placed = False
    pushed_sideways = False
    repositioned = False
    moved_back_2 = False

    cumulative_rot = 0

    obj_positions = {}
    obj_quats = {}

    released = False

    for obj_name in env.sim.model.body_names:
        # Get the position of each object (body) by its name
        obj_pos = env.sim.data.body_xpos[env.sim.model.body_name2id(obj_name)]
        # Get the quaternion (orientation) of the object
        obj_quat = env.sim.data.body_xquat[env.sim.model.body_name2id(obj_name)]
        obj_positions[obj_name] = obj_pos
        obj_quats[obj_name] = obj_quat
        # print(obj_name)
        print(f"Position of {obj_name}: {obj_pos}")

    for rollout_i in tqdm(range(num_rollouts)):
        obs = env.reset()
        for step_i in range(num_steps):

            # sample and execute random action
            # Default:
            # action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1]) 

            # Reach:
            action = reach(obs)

            # Pick:
            # action, has_grasped, move_to_grasp_pose, moved_down, lifted, org_x, org_z = pick(obs, has_grasped, move_to_grasp_pose, moved_down, lifted, org_x, org_z)
            
            # Place: 
            # action, has_grasped, move_to_grasp_pose, moved_down, lifted, placed, org_x, org_z = place(obs, has_grasped, move_to_grasp_pose, moved_down, lifted, placed, org_x, org_z)

            # Pour:
            # action, has_grasped, move_to_grasp_pose, moved_down, lifted, poured, org_x, org_z = pour(obs, has_grasped, move_to_grasp_pose, moved_down, lifted, poured, org_x, org_z)

            # Twist:
            # action, has_grasped, move_to_grasp_pose, moved_down, lifted, poured, org_x, org_z = twist(obs, has_grasped, move_to_grasp_pose, moved_down, lifted, poured, org_x, org_z)  

            # Open:
            # action, has_grasped, move_to_grasp_pose, moved_forward, moved_back, org_y  = open(obs, has_grasped, move_to_grasp_pose, moved_forward, moved_back, org_y)       

            # Close:
            # action, has_grasped, move_to_grasp_pose, moved_forward, pushed_forward, released_grasp, org_y  = close(obs, has_grasped, move_to_grasp_pose, moved_forward, pushed_forward, released_grasp, org_y)         

            # Defrost:
            # action, has_grasped, move_to_grasp_pose, moved_forward, moved_back, moved_down, placed, lifted, door_opened, door_closed, obj_placed, released, pushed_sideways, repositioned, moved_back_2, org_x, org_y, org_z = defrost(obs, obj_positions, obj_quats, has_grasped, 
            #     move_to_grasp_pose, moved_forward, moved_back, moved_down, placed, lifted, door_opened, door_closed, obj_placed, released, pushed_sideways, repositioned, moved_back_2, org_x, org_y, org_z)

            # Place in oven:
            action, has_grasped, move_to_grasp_pose, moved_forward, moved_back, moved_down, placed, lifted, door_opened, door_closed, obj_placed, released, pushed_sideways, repositioned, moved_back_2, cumulative_rot, org_x, org_y, org_z = placeinoven(obs, obj_positions, obj_quats, has_grasped, 
                 move_to_grasp_pose, moved_forward, moved_back, moved_down, placed, lifted, door_opened, door_closed, obj_placed, released, pushed_sideways, repositioned, moved_back_2, cumulative_rot, org_x, org_y, org_z)
            
            if np.allclose(action[:3], [0, 0, 0], atol=1e-4): 
                print("REACHED")
            # else:
                # print('######################################')
                # print(obs)
            obs, _, _, _ = env.step(action)


            """
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
            """
            if video_writer is not None:
                video_img = env.sim.render(
                    # height=512, width=512, camera_name= "robot0_agentview_left"
                    height=512, width=512, camera_name= "robot0_agentview_right"
                    # height=512, width=512, camera_name= "robot0_eye_in_hand"
                )[::-1]
                video_writer.append_data(video_img)

            #if env._check_success():
            #    num_success_rollouts += 1
            #    break

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
