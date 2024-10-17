import robosuite
import numpy as np
from collections import deque

# Drawer: drawer_obj
# SinkFaucet: distr_counter_0
# Coffee: obj
# TurnOnStove: cookware
# Door: door_obj
# CoffeeMug: obj
# Microwave: obj
# TurnSinkSpout: distr_counter_0
# OpenSingleDoor: door_obj

def reach(obs):
    # Extract relevant information from observations
    eef_pos = np.array(obs['robot0_eef_pos'])         # End-effector position
    eef_quat = np.array(obs['robot0_eef_quat'])      # End-effector orientation (quaternion)
    
    # obj_pos = np.array(obs['door_obj_pos'])         # Object position
    # obj_quat = np.array(obs['door_obj_quat'])       # Object orientation (quaternion)

    obj_pos = np.array(obs['obj_pos'])
    obj_quat = np.array(obs['obj_quat'])

    # obj_pos[1] += 0.03
    # obj_pos[2] += 0.12

    # obj_pos[0] -= 0.3
    # obj_pos[1] += 0.04
    # obj_pos[2] -= 0.03

    # Compute position difference (object to end-effector)
    position_diff = obj_pos - eef_pos
    distance = np.linalg.norm(position_diff)

    # Compute rotation difference (object orientation to end-effector orientation)
    rot_diff = robosuite.utils.transform_utils.quat_multiply(obj_quat, robosuite.utils.transform_utils.quat_inverse(eef_quat))
    rot_diff_axis_angle = robosuite.utils.transform_utils.quat2axisangle(rot_diff)

    # Set control gains
    threshold = 0.02  # Distance threshold to stop moving
    Kp_pos = 1  # Position proportional gain
    Kp_rot = 0.00001 # Rotation proportional gain

    # Initialize action array
    action = np.zeros(12)

    # Phase 1: Move toward the object center
    if distance > threshold:
        action[:3] = Kp_pos * position_diff  # Proportional control for position

    # Orientation control (align with object)
    action[3:6] = Kp_rot * rot_diff_axis_angle  # Control for orientation

    # Other controls (gripper, base, torso, wheel)
    action[6:7] = 0 # gripper
    action[7:10] = [0, 0, 0]  # base
    action[10:11] = 0         # torso
    action[11:12] = 0         # wheel (optional)

    return action

def pick(obs, has_grasped=False, moved_to_grasp_pos=False, moved_down=False, lifted=False, org_x=0, org_z=0):
    # Extract relevant information from observations
    eef_pos = np.array(obs['robot0_eef_pos'])         # End-effector position
    eef_quat = np.array(obs['robot0_eef_quat'])       # End-effector orientation (quaternion)
    
    obj_pos = np.array(obs['cookware_pos'])         # Object position
    obj_quat = np.array(obs['cookware_quat'])       # Object orientation (quaternion)

    obj_pos[0] -= 0.02
    obj_pos[1] -= 0.03
    obj_pos[2] += 0.15

    # Compute position difference (object to end-effector)
    position_diff = obj_pos - eef_pos
    distance = np.linalg.norm(position_diff)

    # Compute rotation difference (object orientation to end-effector orientation)
    rot_diff = robosuite.utils.transform_utils.quat_multiply(obj_quat, robosuite.utils.transform_utils.quat_inverse(eef_quat))
    rot_diff_axis_angle = robosuite.utils.transform_utils.quat2axisangle(rot_diff)

    # Set control gains
    threshold = 0.02  # Distance threshold to stop moving
    Kp_pos = 1  # Position proportional gain
    Kp_rot = 0.0001  # Rotation proportional gain

    # Initialize action array
    action = np.zeros(12)

    lift_distance = 0.6  # 1 meter lift distance

    # Phase 1: Move toward the object center
    if not has_grasped:
        if not moved_to_grasp_pos:
            # Move towards the object center first
            if distance > threshold:
                action[:3] = 3 * position_diff  # Proportional control for position
            else:
                # When within the threshold, move 10 cm sideways (x-axis) to prepare for grasp
                moved_to_grasp_pos = True
                org_z = eef_pos[2]
                org_x = eef_pos[1]  # Store current x position for comparison
        else:
            # Phase 2: Move 13 cm sideways along the x-axis
            sideways_distance = 0.02  # 13 cm movement sideways along the x-axis
            sideways_pos = np.array([0, sideways_distance, 0])  # Moving in x-direction (y, x, z)

            action[:3] = 2 * sideways_pos  # Apply the sideways action   

            move_diff = (eef_pos[1] - org_x)  # The sideways movement relative to original position

            if abs(move_diff - sideways_distance) < threshold:
                # Phase 3: Move down before closing the gripper
                move_down_distance = -0.10  # 10 cm movement downward (z-axis)
                move_down_pos = np.array([0, 0, move_down_distance])  # Moving down

                action[:3] = move_down_pos  # Apply the downward action

                move_down_diff = (eef_pos[2] - org_z)  # Check downward movement

                if abs(move_down_diff - move_down_distance) < threshold:
                    # Phase 4: Close the gripper to grasp the object
                    action[6:7] = [1]  # Close the gripper to grasp
                    has_grasped = True  # Mark that the object has been grasped

    # Phase 5: Lift the object after grasping
    elif has_grasped and not lifted:
        # Lift the object by moving upwards (z-axis only)
        lift_target_z = org_z + lift_distance  # Increase only the z-coordinate for lifting
        lift_diff_z = lift_target_z - eef_pos[2]

        # Apply the lifting action only on the z-axis (3rd component of action)
        action[2] = 0.1 * lift_diff_z

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

    return action, has_grasped, moved_to_grasp_pos, moved_down, lifted, org_x, org_z

def place(obs, has_grasped=False, moved_to_grasp_pos=False, moved_down=False, lifted=False, placed=False, moved_back=False, org_x=0, org_y=0, org_z=0, flag = -1):
    # Extract relevant information from observations
    eef_pos = np.array(obs['robot0_eef_pos'])         # End-effector position
    eef_quat = np.array(obs['robot0_eef_quat'])       # End-effector orientation (quaternion)

    # print("eef_quat: ", eef_quat)
    
    obj_pos = np.array(obs['obj_pos'])         # Object position 
    obj_quat = np.array(obs['obj_quat'])       # Object orientation (quaternion)

    if flag == 0:

        # Obj: potato 
        obj_pos[0] += 0.02
        obj_pos[1] -= 0.16
        obj_pos[2] += 0.08

    elif flag == 1:

        obj_pos[2] += 0.05

    # Compute position difference (object to end-effector)
    position_diff = obj_pos - eef_pos
    distance = np.linalg.norm(position_diff)

    # Compute rotation difference (object orientation to end-effector orientation)
    rot_diff = robosuite.utils.transform_utils.quat_multiply(obj_quat, robosuite.utils.transform_utils.quat_inverse(eef_quat))
    rot_diff_axis_angle = robosuite.utils.transform_utils.quat2axisangle(rot_diff)

    # print("rot_diff_angle: ", rot_diff_axis_angle)

    # Set control gains
    threshold = 0.02  # Distance threshold to stop moving
    Kp_pos = 1  # Position proportional gain
    Kp_rot = 0.0001  # Rotation proportional gain

    # Initialize action array
    action = np.zeros(12)


    if flag == 0:
        lift_distance = 0.65
        placement_distance = 0.46
        sideways_distance = 0.15
    elif flag == 1:
        lift_distance =0.2
        placement_distance = -0.65
        sideways_distance = 0
    else:
        lift_distance =0
        placement_distance = 0
        sideways_distance = 0

    rotation = False

    # Phase 1: Move toward the object center
    if not has_grasped:
        if not moved_to_grasp_pos:
            # Move towards the object center first
            if distance > threshold:
                action[:3] = 3 * position_diff  # Proportional control for position
                action[6:7] = -1  # Maintain gripper parallel to the floor or any desired orientation

            else:
                # When within the threshold, move sideways to prepare for grasp
                moved_to_grasp_pos = True
                org_z = eef_pos[2]
                org_x = eef_pos[1]  # Store current x position for comparison
        else:
            # Phase 2: Move sideways along the x-axis
            sideways_pos = np.array([0, sideways_distance, 0])  # Moving in x-direction

            action[:3] = 3 * sideways_pos  # Apply the sideways action   
            move_diff = (eef_pos[1] - org_x)  # The sideways movement relative to original position

            if flag == 0:
                action[3:6] = 0.4 * np.array([0, 0, 0.5])
                rotation = True
            elif flag == 1:
                action[3:6] = 0.7 * np.array([0, 0, 0.5])
                rotation = True

            if abs(move_diff - sideways_distance) < threshold:
                # Phase 3: Move down before closing the gripper

                if flag == 0:
                    move_down_distance = -0.01
                elif flag == 1:
                # move_down_distance = -0.1  # 9 cm movement downward (z-axis)
                    move_down_distance = -0.08
                else:
                    move_down_distance = 0
                move_down_pos = np.array([0.02, 0, move_down_distance])  # Moving down

                action[:3] = 5 * move_down_pos  # Apply the downward action

                action[6:7] = -1  # Open the gripper slightly

                move_down_diff = (eef_pos[2] - org_z)  # Check downward movement

                print("TURN")

                rotation = True

                # Apply the rotation to the action
                if flag == 0:
                    action[3:6] = 0.4 * np.array([0, 0, 0.5])
                    rotation = True
                elif flag == 1:
                    action[3:6] = 0.7 * np.array([0, 0, 0.5])
                    rotation = True

                if abs(move_down_diff - move_down_distance) < threshold:
                    
                    has_grasped = True  # Mark that the object has been grasped (optional if grasping)
                    action[6:7] = 1
                    org_z = eef_pos[2]

    # Phase 5: Lift the object after grasping
    elif has_grasped and not lifted:
        # Lift the object by moving upwards (z-axis only)
        lift_target_z = org_z + lift_distance  # Increase only the z-coordinate for lifting
        lift_diff_z = lift_target_z - eef_pos[2]
        print("LIFT DIFF: ", lift_diff_z)

        if lift_diff_z > 0.63:
            # Apply the lifting action only on the z-axis (3rd component of action)
            action[2] = 0.1 * lift_diff_z
        else:
            # Apply the lifting action only on the z-axis (3rd component of action)
            action[2] = 3 * lift_diff_z

        # Keep the gripper closed while lifting
        action[6:7] = [1]  # Keep gripper closed

        # Check if the robot has lifted the object enough
        if abs(lift_diff_z) < threshold:
            lifted = True  # Mark that the object has been lifted
            org_y = eef_pos[0]

    # Phase 6: Place the object
    elif lifted and not placed:

        # Phase 5: Move backwards before placing the object
        if not moved_back:
            # Define the distance to move back (negative x direction)

            if flag == 0:
                backward_distance = -0.2
            else:
            # backward_distance = -0.2  # e.g., 10 cm backward
                backward_distance = 0
            move_back_pos = np.array([backward_distance, 0, 0])  # Move backward only along x-axis

            # Apply the backward action
            action[:3] = 10 * move_back_pos  # Apply a scaled backward movement

            # Check if the robot has moved back to the desired position
            back_diff = eef_pos[0] - (org_y)  # The movement along the x-axis relative to the original position
            action[6:7] = [1]  # Keep gripper closed
            if abs(back_diff - backward_distance) < threshold:
                moved_back = True  # Mark that the robot has successfully moved back
                org_x = eef_pos[1]
                org_y = eef_pos[0]
                org_z = eef_pos[2]

        # After moving back, proceed to the placement phase
        if moved_back:
            # Move horizontally to the target placement position
            place_target_pos = np.array([0, placement_distance, 0])  # Move horizontally
            action[:3] = 50 * place_target_pos  # Apply the sideways action   
            place_diff = (eef_pos[1] - (org_x))  # The sideways movement relative to original position
            print("PLACE_DIFF: ", place_diff)

            # Check if the robot has moved to the placement position
            if abs(place_diff - placement_distance) < threshold:

                if flag == 0:
                    place_down_distance = 0.35
                    place_down_pos = np.array([place_down_distance, 0, 0])
                elif flag == 1:
                    place_down_distance = -0.14
                    place_down_pos = np.array([0, 0, place_down_distance])
                else:
                    place_down_distance = 0
                    place_down_pos = np.array([0, 0, place_down_distance])

                # place_down_pos = np.array([place_down_distance, 0, 0]

                action[:3] = 50 * place_down_pos  # Apply the downward action

                place_down_diff = (eef_pos[2] - org_z)  # Check downward movement

                print("MOVE FORWARD: ", place_down_diff)

                if (abs(place_down_distance) > 0.06):
                    action[6:7] = -2
                

                # if abs(place_down_diff - place_down_distance) < 0.01:
                if abs(place_down_diff - place_down_distance) < threshold:
                    print("placed")
                    # Open the gripper to release the object

                    if flag == 0:
                        down = -0.05

                    else:
                        down = 0
                    # down = -0.05
                    down = 0
                    if down == 0:
                        placed = True
                    action[:3] = 50 * np.array([0,0,down])
                    action[6:7] = [-2]  # Open the gripper to release the object
                    if abs(eef_pos[2] - org_z - down) < threshold:
                        placed = True  # Mark that the object has been placed

    if not rotation:
        # Orientation control (align with object)
        action[3:6] = Kp_rot * rot_diff_axis_angle  # Control for orientation

    # Other controls (base, torso, wheel)
    action[7:10] = [0, 0, 0]  # base
    action[10:11] = 0         # torso
    action[11:12] = 0         # wheel (optional)

    return action, has_grasped, moved_to_grasp_pos, moved_down, lifted, placed, moved_back, org_x, org_y, org_z

def pour(obs, has_grasped=False, moved_to_grasp_pos=False, moved_down=False, lifted=False, poured=False, org_x=0, org_z=0):
    # Extract relevant information from observations
    eef_pos = np.array(obs['robot0_eef_pos'])         # End-effector position
    eef_quat = np.array(obs['robot0_eef_quat'])       # End-effector orientation (quaternion)
    
    obj_pos = np.array(obs['cookware_pos'])         # Object position
    obj_quat = np.array(obs['cookware_quat'])       # Object orientation (quaternion)

    # obj_pos[0] -= 0.02
    # obj_pos[1] -= 0.03
    # obj_pos[2] += 0.15

    # Compute position difference (object to end-effector)
    position_diff = obj_pos - eef_pos
    distance = np.linalg.norm(position_diff)

    # Compute rotation difference (object orientation to end-effector orientation)
    rot_diff = robosuite.utils.transform_utils.quat_multiply(obj_quat, robosuite.utils.transform_utils.quat_inverse(eef_quat))
    rot_diff_axis_angle = robosuite.utils.transform_utils.quat2axisangle(rot_diff)

    # Set control gains
    threshold = 0.02  # Distance threshold to stop moving
    Kp_pos = 1  # Position proportional gain
    Kp_rot = 0.0001  # Rotation proportional gain

    # Initialize action array
    action = np.zeros(12)

    lift_distance = 0.2  # 1 meter lift distance
    pour_angle = 0.75  # Target pour angle in radians (adjust this for the desired tilt)

    # Phase 1: Move toward the object center
    if not has_grasped:
        if not moved_to_grasp_pos:
            # Move towards the object center first
            if distance > threshold:
                action[:3] = 3 * position_diff  # Proportional control for position
            else:
                # When within the threshold, move 10 cm sideways (x-axis) to prepare for grasp
                moved_to_grasp_pos = True
                org_z = eef_pos[2]
                org_x = eef_pos[1]  # Store current x position for comparison
        else:
            # Phase 2: Move 13 cm sideways along the x-axis
            sideways_distance = 0.13  # 13 cm movement sideways along the x-axis
            sideways_pos = np.array([0, sideways_distance, 0])  # Moving in x-direction (y, x, z)

            action[:3] = 2 * sideways_pos  # Apply the sideways action   

            move_diff = (eef_pos[1] - org_x)  # The sideways movement relative to original position

            if abs(move_diff - sideways_distance) < threshold:
                # Phase 3: Move down before closing the gripper
                move_down_distance = -0.03  # 10 cm movement downward (z-axis)
                move_down_pos = np.array([0, 0, move_down_distance])  # Moving down

                action[:3] = move_down_pos  # Apply the downward action

                move_down_diff = (eef_pos[2] - org_z)  # Check downward movement

                if abs(move_down_diff - move_down_distance) < threshold:
                    # Phase 4: Close the gripper to grasp the object
                    action[6:7] = [1]  # Close the gripper to grasp
                    has_grasped = True  # Mark that the object has been grasped

    # Phase 5: Lift the object after grasping
    elif has_grasped and not lifted:
        # Lift the object by moving upwards (z-axis only)
        lift_target_z = org_z + lift_distance  # Increase only the z-coordinate for lifting
        lift_diff_z = lift_target_z - eef_pos[2]

        if lift_diff_z > 0.20:
            action[2] = 0.1 * lift_diff_z  # Apply the lifting action only on the z-axis
        else:
            action[2] = 3 * lift_diff_z

        # Keep the gripper closed while lifting
        action[6:7] = [1]  # Keep gripper closed

        # Check if the robot has lifted the object enough
        if abs(lift_diff_z) < threshold:
            lifted = True  # Mark that the object has been lifted

    # Phase 6: Pour the object after lifting
    elif lifted and not poured:
        # Define the pour rotation (rotating around the y-axis to simulate pouring)
        pour_rotation = np.array([0, pour_angle, 0])  # Rotation around the y-axis

        # Apply the pouring action by changing the orientation of the end-effector
        action[3:6] = 0.1 * pour_rotation  # Apply proportional control for rotation

        # Once the desired pour angle is reached, mark as poured
        if abs(rot_diff_axis_angle[1] - pour_angle) < threshold:
            poured = True  # Mark that the object has been poured

    if not lifted or poured:
        # Orientation control (align with object)
        action[3:6] = Kp_rot * rot_diff_axis_angle  # Control for orientation

    # Other controls (base, torso, wheel)
    action[7:10] = [0, 0, 0]  # base
    action[10:11] = 0         # torso
    action[11:12] = 0         # wheel (optional)

    return action, has_grasped, moved_to_grasp_pos, moved_down, lifted, poured, org_x, org_z

def twist(obs, has_grasped=False, moved_to_grasp_pos=False, moved_down=False, lifted=False, poured=False, org_x=0, org_z=0):
    # Extract relevant information from observations
    eef_pos = np.array(obs['robot0_eef_pos'])         # End-effector position
    eef_quat = np.array(obs['robot0_eef_quat'])       # End-effector orientation (quaternion)
    
    obj_pos = np.array(obs['cookware_pos'])         # Object position
    obj_quat = np.array(obs['cookware_quat'])       # Object orientation (quaternion)

    # Compute position difference (object to end-effector)
    position_diff = obj_pos - eef_pos
    distance = np.linalg.norm(position_diff)

    # Compute rotation difference (object orientation to end-effector orientation)
    rot_diff = robosuite.utils.transform_utils.quat_multiply(obj_quat, robosuite.utils.transform_utils.quat_inverse(eef_quat))
    rot_diff_axis_angle = robosuite.utils.transform_utils.quat2axisangle(rot_diff)

    # Set control gains
    threshold = 0.02  # Distance threshold to stop moving
    Kp_pos = 1  # Position proportional gain
    Kp_rot = 0.00001  # Increased Rotation proportional gain

    # Initialize action array
    action = np.zeros(12)

    lift_distance = 0.2  # 20 cm lift distance
    twist_angle = 1.2  # Target twist angle in radians for pouring

    # Phase 1: Move toward the object center
    if not has_grasped:
        if not moved_to_grasp_pos:
            # Move towards the object center first
            if distance > threshold:
                action[:3] = 3 * position_diff  # Proportional control for position
            else:
                # When within the threshold, move 10 cm sideways (x-axis) to prepare for grasp
                moved_to_grasp_pos = True
                org_z = eef_pos[2]
                org_x = eef_pos[1]  # Store current x position for comparison
        else:
            # Phase 2: Move 13 cm sideways along the x-axis
            sideways_distance = 0.13  # 13 cm movement sideways along the x-axis
            sideways_pos = np.array([0, sideways_distance, 0])  # Moving in x-direction (y, x, z)

            action[:3] = 2 * sideways_pos  # Apply the sideways action   

            move_diff = (eef_pos[1] - org_x)  # The sideways movement relative to original position

            if abs(move_diff - sideways_distance) < threshold:
                # Phase 3: Move down before closing the gripper
                move_down_distance = -0.03  # 3 cm movement downward (z-axis)
                move_down_pos = np.array([0, 0, move_down_distance])  # Moving down

                action[:3] = move_down_pos  # Apply the downward action

                move_down_diff = (eef_pos[2] - org_z)  # Check downward movement

                if abs(move_down_diff - move_down_distance) < threshold:
                    # Phase 4: Close the gripper to grasp the object
                    action[6:7] = [1]  # Close the gripper to grasp
                    has_grasped = True  # Mark that the object has been grasped

    # Phase 5: Lift the object after grasping
    elif has_grasped and not lifted:
        # Lift the object by moving upwards (z-axis only)
        lift_target_z = org_z + lift_distance  # Increase only the z-coordinate for lifting
        lift_diff_z = lift_target_z - eef_pos[2]

        if lift_diff_z > 0.20:
            action[2] = 0.1 * lift_diff_z  # Apply the lifting action only on the z-axis
        else:
            action[2] = 3 * lift_diff_z

        # Keep the gripper closed while lifting
        action[6:7] = [1]  # Keep gripper closed

        # Check if the robot has lifted the object enough
        if abs(lift_diff_z) < threshold:
            lifted = True  # Mark that the object has been lifted

    # Phase 6: Twist and pour the object after lifting
    elif lifted and not poured:
        # Define the twist rotation (rotating around the x-axis to simulate pouring)
        twist_axis = np.array([1, 0, 0])  # Rotation around the x-axis
        desired_quat = robosuite.utils.transform_utils.axisangle2quat(twist_axis * twist_angle)

        # Compute the rotational difference between the current and desired orientation
        rot_diff = robosuite.utils.transform_utils.quat_multiply(desired_quat, robosuite.utils.transform_utils.quat_inverse(eef_quat))
        rot_diff_axis_angle = robosuite.utils.transform_utils.quat2axisangle(rot_diff)

        # Apply the twist (rotation) action
        action[3:6] = 0.1 * rot_diff_axis_angle  # Apply the twist with proportional control

        # Once the desired pour angle (twist) is reached, mark as poured
        if abs(rot_diff_axis_angle[0] - twist_angle) < threshold:
            poured = True  # Mark that the object has been poured

    if not lifted or poured:
        # Orientation control (align with object)
        action[3:6] = Kp_rot * rot_diff_axis_angle  # Control for orientation

    # Other controls (base, torso, wheel)
    action[7:10] = [0, 0, 0]  # base
    action[10:11] = 0         # torso
    action[11:12] = 0         # wheel (optional)

    return action, has_grasped, moved_to_grasp_pos, moved_down, lifted, poured, org_x, org_z

def open(obs, has_grasped=False, moved_to_grasp_pos=False, moved_forward=False, moved_back=False, released=False, pushed_sideways=False, 
         repositioned=False, moved_back_2=False, org_x=0, org_y=0, target_pos=np.array([-1,-1,-1]), target_quat=np.array([-1,-1])):
    # Extract relevant information from observations
    eef_pos = np.array(obs['robot0_eef_pos'])         # End-effector position
    eef_quat = np.array(obs['robot0_eef_quat'])       # End-effector orientation (quaternion)

    if target_pos.all() == -1:
        obj_pos = np.array(obs['door_obj_pos'])           # Object position
        obj_quat = np.array(obs['door_obj_quat'])         # Object orientation (quaternion)
    else:
        obj_pos = target_pos
        obj_quat = target_quat

    # Adjust object position if needed to align better with the handle
    obj_pos[0] -= 0.3
    obj_pos[1] -= 0.201

    # Compute position difference (object to end-effector)
    position_diff = obj_pos - eef_pos
    distance = np.linalg.norm(position_diff)

    # Compute rotation difference (object orientation to end-effector orientation)
    rot_diff = robosuite.utils.transform_utils.quat_multiply(obj_quat, robosuite.utils.transform_utils.quat_inverse(eef_quat))
    rot_diff_axis_angle = robosuite.utils.transform_utils.quat2axisangle(rot_diff)

    # Set control gains
    threshold = 0.02  # Distance threshold to stop moving
    Kp_pos = 1        # Position proportional gain
    Kp_rot = 0.00001  # Rotation proportional gain

    # Initialize action array
    action = np.zeros(12)

    # Desired orientation (parallel to the floor)
    desired_quat = np.array([0, 0, 1, 0])  # Example quaternion (no rotation around x-axis)

    # Compute the rotational difference between current and desired orientation
    parallel_rot_diff = robosuite.utils.transform_utils.quat_multiply(desired_quat, robosuite.utils.transform_utils.quat_inverse(eef_quat))
    parallel_rot_diff_axis_angle = robosuite.utils.transform_utils.quat2axisangle(parallel_rot_diff)

    # Phase 1: Move toward the object center
    if not has_grasped:
        if not moved_to_grasp_pos:
            # Move towards the object center (x, y, z direction separately)
            if distance > threshold:
                action[:3] = 3 * position_diff  # Proportional control for position
                action[6:7] = 0
            else:
                # When close enough, mark that we reached the grasp position
                moved_to_grasp_pos = True
                org_y = eef_pos[0]  # Store original y position to compare later
                org_x = eef_pos[1]

        elif not moved_forward:
            # Phase 2: Move forward a bit (y-direction)
            forward_distance = 0.08  # Move forward 8 cm to grasp handle
            forward_pos = np.array([forward_distance, 0, 0])
            action[:3] = forward_pos  # Move forward along x-axis
            action[6:7] = -0.3

            # Check if we have moved forward enough
            if abs(eef_pos[0] - org_y - forward_distance) < threshold:
                moved_forward = True
                org_y = eef_pos[0]
                action[6:7] = 1

        else:
            # Phase 3: Grasp the handle by closing the gripper
            action[6:7] = 2  # Close the gripper to grasp the handle
            has_grasped = True  # Mark that we have grasped the handle

    elif has_grasped and not moved_back:
        # Move backward diagonally (along x and y axes)
        back_distance_x = 0.23  # distance to move back along the x-axis
        back_distance_y = 0.23  # distance to move diagonally along the y-axis

        # Calculate the diagonal direction
        move_vector_x = back_distance_x  # negative for moving backward along x-axis
        move_vector_y = -1 * back_distance_y  # diagonal shift along the y-axis

        action[6:7] = 2

        if abs(eef_pos[1] - org_x) < 0.05 and abs(eef_pos[0] - org_y) < 0.05:
            action[1] = move_vector_x  # move back along the x-axis
            action[0] = move_vector_y  # move diagonally along the y-axis
        else:
            action[1] = 2 * move_vector_x  # Move back along the x-axis
            action[0] = 2 * move_vector_y  # Move diagonally along the y-axis

        print("XABS: ", eef_pos[1] - org_x - back_distance_x)
        print("YABS: ", eef_pos[0] - org_y + back_distance_y)

        # Check if the robot has moved back enough (diagonally)
        if abs(eef_pos[1] - org_x - back_distance_x) < 0.1 and abs(eef_pos[0] - org_y + back_distance_y) < 0.1:
            moved_back = True
            org_y = eef_pos[0]

    elif moved_back and not released:
        # Phase 4: Release the handle by opening the gripper
        print("Releasing the handle")
        action[6:7] = -2  # Open the gripper to release the handle
        # Phase 5: Move back an additional 2 cm
        print("Moving back after release")
        move_back_dist = -0.13  # Move back 2 cm in the x-axis
        action[:3] = 5 * np.array([move_back_dist, 0.07, 0])
        action[6:7] = -1

        # Check if the robot has moved back 2 cm
        if abs(eef_pos[0] - org_y - move_back_dist) < threshold:
            print("Move back completed")
            released = True  # Mark that we have released the handle
            org_x = eef_pos[1]
            org_y = eef_pos[0]
            moved_back = False

        # Intermediate Phase: Reposition the robot arm before pushing
    if released and not repositioned:
        print("REPOSITION")
        # Move the arm up slightly and away from the door to create room for the push
        reposition_distance = -0.17  # Distance to move in the z direction (upward)
        retreat_distance = 0.1    # Distance to move back along the x-axis
        reposition_vector = np.array([retreat_distance, reposition_distance, 0])  # Move back and up

        action[:3] = reposition_vector  # Apply the repositioning movement
        print("Y: eef_pos[0] - org_y")

        # Check if the robot has moved enough to reposition
        if abs(eef_pos[0] - org_y - retreat_distance) < threshold and abs(eef_pos[1] - org_x - reposition_distance) < threshold:
            repositioned = True  # Mark the repositioning as complete
            print("Repositioning completed")

        # New Phase: Sideways push to open the door
    if repositioned and not pushed_sideways:
        # Determine the direction to push (e.g., along the x-axis to the side)
        side_push_distance = 0.2  # Adjust distance as needed
        side_push_direction = np.array([0, side_push_distance, 0])  # Push along x-axis

        # Apply force to push the door open
        action[:3] = 5 * side_push_direction

        # Check if the robot has pushed far enough
        if abs(eef_pos[1] - org_x - side_push_distance) < threshold:
            pushed_sideways = True  # Mark the push as complete
            print("Pushed the door open sideways")
            org_y = eef_pos[0]

    if pushed_sideways and not moved_back_2:
        # Phase 4: Release the handle by opening the gripper
        print("Releasing the handle")
        # Phase 5: Move back an additional 2 cm
        print("Moving back after release")
        move_back_dist = -0.25  # Move back 2 cm in the x-axis
        action[0] = 5 * move_back_dist

        # Check if the robot has moved back 2 cm
        if abs(eef_pos[0] - org_y - move_back_dist) < threshold:
            print("Move back completed")
            moved_back_2 = True

    # Maintain orientation and return action
    action[3:6] = Kp_rot * robosuite.utils.transform_utils.quat2axisangle(
        robosuite.utils.transform_utils.quat_multiply(
            np.array([0, 0, 1, 0]), robosuite.utils.transform_utils.quat_inverse(eef_quat)
        )
    )

    # Orientation control (keep gripper parallel to the floor)
    action[3:6] = Kp_rot * parallel_rot_diff_axis_angle  # Control for maintaining parallel orientation

    # Other controls (base, torso, wheel)
    action[7:10] = [0, 0, 0]  # base
    action[10:11] = 0         # torso
    action[11:12] = 0         # wheel (optional)

    return action, has_grasped, moved_to_grasp_pos, moved_forward, moved_back, released, pushed_sideways, repositioned, moved_back_2, org_x, org_y

def close(obs, has_grasped=False, moved_to_grasp_pos=False, moved_forward=False, pushed_forward=False, released_grasp=False, moved_back=False, org_x=0, org_y=0, target_pos=np.array([-1,-1,-1]), target_quat=np.array([-1,-1])):
    print("CLOSE")
    # Extract relevant information from observations
    eef_pos = np.array(obs['robot0_eef_pos'])         # End-effector position
    eef_quat = np.array(obs['robot0_eef_quat'])       # End-effector orientation (quaternion)
    
    if target_pos.all() == -1:
        obj_pos = np.array(obs['door_obj_pos'])           # Object position
        obj_quat = np.array(obs['door_obj_quat'])         # Object orientation (quaternion)
    else:
        obj_pos = target_pos
        obj_quat = target_quat

    # Adjust object position if needed to align better with the handle
    # obj_pos[0] -= 0.8
    # obj_pos[1] += 0.63
    # obj_pos[2] += 0.2

    # obj_pos[1] -= 0.18
    # obj_pos[0] -= 0.2
    # obj_pos[2] += 0.05

    # Compute position difference (object to end-effector)
    position_diff = eef_pos - obj_pos
    distance = np.linalg.norm(position_diff)

    # Compute rotation difference (object orientation to end-effector orientation)
    rot_diff = robosuite.utils.transform_utils.quat_multiply(obj_quat, robosuite.utils.transform_utils.quat_inverse(eef_quat))
    rot_diff_axis_angle = robosuite.utils.transform_utils.quat2axisangle(rot_diff)

    # Set control gains
    threshold = 0.02  # Distance threshold to stop moving
    Kp_pos = 1        # Position proportional gain
    Kp_rot = 0.000001  # Rotation proportional gain

    # Initialize action array
    action = np.zeros(12)

    # Phase 1: Move toward the object center
    if not has_grasped:
        print("grasp")
        print("moved_back: ", moved_back)

        if not moved_to_grasp_pos:
            """
            # Move towards the object center (x, y, z direction separately)
            print("DIST: ", distance)
            if distance > threshold:
                action[:3] = 3 * position_diff  # Proportional control for position
            else:
                # When close enough, mark that we reached the grasp position
                moved_to_grasp_pos = True
                org_y = eef_pos[0]  # Store original y position to compare later
            """
            print("11111111")
            org_y = eef_pos[0]
            moved_to_grasp_pos = True
        
        if moved_to_grasp_pos and not moved_back:
            print("222222")
            back_distance = -0.98 # Move forward 10 cm
            back_pos = np.array([back_distance, 0, 0])
            action[:3] = 50 * back_pos  # Move forward along x-axis
            print("BACK: ", eef_pos[0] - org_y - back_distance)

            # Check if we have moved forward enough
            if abs(eef_pos[0] - org_y - back_distance) < threshold:
                print("MOVED_BACK")
                moved_back = True
                org_y = eef_pos[0]
                org_x = eef_pos[1]

        if moved_back and not moved_forward:
            print("3333333")
            print("FORWARD")
            # Phase 2: Move forward a bit before grasping
            forward_distance = 0.17  # Move forward 10 cm
            forward_pos = np.array([forward_distance, 0.4, 0])
            action[:3] = 5 * forward_pos  # Move forward along x-axis

            # Check if we have moved forward enough
            if abs(eef_pos[1] - org_x - 0.4) < threshold:
                moved_forward = True
                org_x = eef_pos[1]
                org_y = eef_pos[0]

        if moved_forward:
            print("444444")
            # Phase 3: Move sideways instead of grasping the handle
            sideways_distance = -0.3  # Distance to move sideways (along the y-axis)
            sideways_pos = np.array([0, sideways_distance, 0])  # Move along the y-axis
            action[:3] = 5 * sideways_pos  # Proportional control for sideways movement

            if abs(eef_pos[1] - org_x - sideways_distance) < threshold:

                forward_dist = 0.3
                forward_position = np.array([forward_dist, 0, 0])
                action[:3] = 5  * forward_position

                if abs(eef_pos[0] - org_y - forward_dist) < threshold:
                
                    # Update state to indicate that the movement is complete
                    has_grasped = True  # Mark that the sideways move is complete (replace this with a better flag if needed)

    # Phase 4: Push forward to close the door
    elif has_grasped and not pushed_forward:
        # Move forward along the x-axis to push the handle
        forward_distance = 0.20  # Push 20 cm forward
        action[0] = Kp_pos * forward_distance  # Move forward along the x-axis
        action[6:7] = 1  # Keep the gripper closed during the push

        # Check if the robot has pushed forward enough
        if abs(eef_pos[0] - org_y - forward_distance) < threshold:
            pushed_forward = True

    # Phase 5: Release the handle
    elif pushed_forward and not released_grasp:
        action[6:7] = -1  # Open the gripper to release the handle
        released_grasp = True

    # Orientation control (keep gripper parallel to the floor)
    action[3:6] = 0.00001 * rot_diff_axis_angle  # Control for maintaining parallel orientation

    # Other controls (base, torso, wheel)
    action[7:10] = [0, 0, 0]  # base
    action[10:11] = 0         # torso
    action[11:12] = 0         # wheel (optional)

    return action, has_grasped, moved_to_grasp_pos, moved_forward, pushed_forward, released_grasp, moved_back, org_x, org_y

def press(obs):
    # Extract relevant information from observations
    eef_pos = np.array(obs['robot0_eef_pos'])         # End-effector position
    eef_quat = np.array(obs['robot0_eef_quat'])       # End-effector orientation (quaternion)

    obj_pos = np.array(obs['button_obj_pos'])         # Object position (assuming a button)
    obj_quat = np.array(obs['button_obj_quat'])       # Object orientation (quaternion)

    # Adjust object position for pressing (move along z-axis)
    press_distance = -0.05  # Set a small distance to press downward (negative z-axis)
    press_axis = np.array([0, 0, 1])  # Pressing along the z-axis
    obj_pos += press_distance * press_axis  # Move object position downward

    # Compute position difference (target press position to end-effector position)
    position_diff = obj_pos - eef_pos
    distance = np.linalg.norm(position_diff)

    # Compute rotation difference (object orientation to end-effector orientation)
    rot_diff = robosuite.utils.transform_utils.quat_multiply(obj_quat, robosuite.utils.transform_utils.quat_inverse(eef_quat))
    rot_diff_axis_angle = robosuite.utils.transform_utils.quat2axisangle(rot_diff)

    # Set control gains
    threshold = 0.01  # Distance threshold to stop moving (closer threshold for pressing)
    Kp_pos = 1  # Position proportional gain
    Kp_rot = 0.00001  # Rotation proportional gain

    # Initialize action array
    action = np.zeros(12)

    # Phase 1: Press the button by moving along the z-axis (downward)
    if distance > threshold:
        action[:3] = Kp_pos * position_diff  # Proportional control for position

    # Orientation control (align with object)
    action[3:6] = Kp_rot * rot_diff_axis_angle  # Control for orientation

    # Other controls (gripper, base, torso, wheel)
    action[6:7] = 0  # Gripper remains open during press (can modify if needed)
    action[7:10] = [0, 0, 0]  # base (if applicable)
    action[10:11] = 0         # torso (if applicable)
    action[11:12] = 0         # wheel (optional)

    return action

def openoven(obs, has_grasped=False, moved_to_grasp_pos=False, moved_forward=False, moved_back=False, released=False, pushed_sideways=False, 
         repositioned=False, moved_back_2=False, cumulative_rot = 0, org_x=0, org_y=0, target_pos=np.array([-1,-1,-1]), target_quat=np.array([-1,-1])):
    # Extract relevant information from observations
    eef_pos = np.array(obs['robot0_eef_pos'])         # End-effector position
    eef_quat = np.array(obs['robot0_eef_quat'])       # End-effector orientation (quaternion)

    if target_pos.all() == -1:
        obj_pos = np.array(obs['door_obj_pos'])           # Object position
        obj_quat = np.array(obs['door_obj_quat'])         # Object orientation (quaternion)
    else:
        obj_pos = target_pos
        obj_quat = target_quat

    obj_pos[0] -= 0.08
    obj_pos[1] -= 0.1
    obj_pos[2] -= 0.0

    # Adjust object position if needed to align better with the handle

    # Compute position difference (object to end-effector)
    position_diff = obj_pos - eef_pos
    distance = np.linalg.norm(position_diff)

    # Compute rotation difference (object orientation to end-effector orientation)
    rot_diff = robosuite.utils.transform_utils.quat_multiply(obj_quat, robosuite.utils.transform_utils.quat_inverse(eef_quat))
    rot_diff_axis_angle = robosuite.utils.transform_utils.quat2axisangle(rot_diff)

    # Set control gains
    threshold = 0.02  # Distance threshold to stop moving
    Kp_pos = 1        # Position proportional gain
    Kp_rot = 0.00001  # Rotation proportional gain

    # Initialize action array
    action = np.zeros(12)

    # Compute rotation difference (object orientation to end-effector orientation)
    rot_diff = robosuite.utils.transform_utils.quat_multiply(obj_quat, robosuite.utils.transform_utils.quat_inverse(eef_quat))
    rot_diff_axis_angle = robosuite.utils.transform_utils.quat2axisangle(rot_diff)

    rotation = True


    # Phase 1: Move toward the object center
    if not has_grasped:
        print("grasp")
        print("moved_back: ", moved_back)

        if not moved_to_grasp_pos:
            """
            # Move towards the object center (x, y, z direction separately)
            print("DIST: ", distance)
            if distance > threshold:
                action[:3] = 3 * position_diff  # Proportional control for position
            else:
                # When close enough, mark that we reached the grasp position
                moved_to_grasp_pos = True
                org_y = eef_pos[0]  # Store original y position to compare later
            """
            print("11111111")
            org_y = eef_pos[0]
            moved_to_grasp_pos = True
        
        if moved_to_grasp_pos and not moved_back:
            print("222222")
            back_distance = -0.20 # Move forward 10 cm
            back_pos = np.array([back_distance, 0, 0.0])
            action[:3] = 50 * back_pos  # Move forward along x-axis
            print("EEF: ", eef_pos[0])
            print("ORG: ", org_y)
            print("BACK: ", eef_pos[0] - org_y)

            # Check if we have moved forward enough
            if abs(eef_pos[0] - org_y - back_distance) < threshold:
                print("MOVED_BACK")
                moved_back = True
                org_y = eef_pos[0]
                org_x = eef_pos[2]

        elif not moved_forward:
            print("move_forward")
            # Phase 2: Move forward a bit (y-direction)
            down_distance = -0.18  # Move forward 8 cm to grasp handle
            down_pos = np.array([0.03, 0.05, down_distance])
            action[:3] = 3 * down_pos  # Move forward along x-axis
            action[6:7] = -0.3
            action[3:6] = 0.8 * np.array([0, 0, -0.15])
            rotation = True

            # Check if we have moved forward enough
            if abs(eef_pos[2] - org_x - down_distance) < threshold:
                moved_forward = True
                org_y = eef_pos[0]
                action[6:7] = 0

        else:
            # Phase 3: Grasp the handle by closing the gripper
            action[6:7] = 0  # Close the gripper to grasp the handle
            has_grasped = True  # Mark that we have grasped the handle
            cumulative_rot = 0

    elif has_grasped and not moved_back:
        # Move backward diagonally (along x and y axes)
        back_distance_y = -0.10  # distance to move diagonally along the y-axis

        move_vector_y = -1 * back_distance_y  # diagonal shift along the y-axis

        action[6:7] = 0

        action[0] = 2 * move_vector_y  # Move diagonally along the y-axis
        # action[3:6] = 0.7 * np.array([0, 0, -0.5])
        rotation = True

        print("YABS: ", eef_pos[0] - org_y + back_distance_y)

        # Check if the robot has moved back enough (diagonally)
        if abs(eef_pos[0] - org_y + back_distance_y) < 0.01:
            moved_back = True
            org_y = eef_pos[0]

    elif moved_back and not released:
        # Phase 4: Release the handle by opening the gripper

        step = 0.1

        print("STEP")

        action[0:3] = 0.07* np.array([-0.02, 0, 0])
        action[6:7] = 2
        action[3:6] = 0.7 * np.array([-0.5, 0, 0])

        cumulative_rot += step

        print("CUMULATIVE: ", cumulative_rot)
        rotation = True

        if cumulative_rot > 1:
            released = True
        # Check if the robot has moved back 2 cm
    if not rotation:
        # Orientation control (keep gripper parallel to the floor)
        action[3:6] = Kp_rot * rot_diff_axis_angle  # Control for maintaining parallel orientation

    # Other controls (base, torso, wheel)
    action[7:10] = [0, 0, 0]  # base
    action[10:11] = 0         # torso
    action[11:12] = 0         # wheel (optional)

    return action, has_grasped, moved_to_grasp_pos, moved_forward, moved_back, released, pushed_sideways, repositioned, moved_back_2, cumulative_rot, org_x, org_y
    # Extract relevant information from observations
    eef_pos = np.array(obs['robot0_eef_pos'])         # End-effector position
    eef_quat = np.array(obs['robot0_eef_quat'])       # End-effector orientation (quaternion)

    obj_pos = np.array(obs['button_obj_pos'])         # Object position (assuming a button)
    obj_quat = np.array(obs['button_obj_quat'])       # Object orientation (quaternion)

    # Adjust object position for pressing (move along z-axis)
    press_distance = -0.05  # Set a small distance to press downward (negative z-axis)
    press_axis = np.array([0, 0, 1])  # Pressing along the z-axis
    obj_pos += press_distance * press_axis  # Move object position downward

    # Compute position difference (target press position to end-effector position)
    position_diff = obj_pos - eef_pos
    distance = np.linalg.norm(position_diff)

    # Compute rotation difference (object orientation to end-effector orientation)
    rot_diff = robosuite.utils.transform_utils.quat_multiply(obj_quat, robosuite.utils.transform_utils.quat_inverse(eef_quat))
    rot_diff_axis_angle = robosuite.utils.transform_utils.quat2axisangle(rot_diff)

    # Set control gains
    threshold = 0.01  # Distance threshold to stop moving (closer threshold for pressing)
    Kp_pos = 1  # Position proportional gain
    Kp_rot = 0.00001  # Rotation proportional gain

    # Initialize action array
    action = np.zeros(12)

    # Phase 1: Press the button by moving along the z-axis (downward)
    if distance > threshold:
        action[:3] = Kp_pos * position_diff  # Proportional control for position

    # Orientation control (align with object)
    action[3:6] = Kp_rot * rot_diff_axis_angle  # Control for orientation

    # Other controls (gripper, base, torso, wheel)
    action[6:7] = 0  # Gripper remains open during press (can modify if needed)
    action[7:10] = [0, 0, 0]  # base (if applicable)
    action[10:11] = 0         # torso (if applicable)
    action[11:12] = 0         # wheel (optional)

    return action


def push(obs, org_x=0):
    # Extract relevant information from observations
    eef_pos = np.array(obs['robot0_eef_pos'])         # End-effector position
    eef_quat = np.array(obs['robot0_eef_quat'])       # End-effector orientation (quaternion)

    obj_pos = np.array(obs['door_obj_pos'])           # Object position (assuming door or any object)
    obj_quat = np.array(obs['door_obj_quat'])         # Object orientation (quaternion)

    # Adjust object position for the push direction (e.g., push along x-axis)
    push_distance = 0.2  # Set a distance to push along x-axis
    push_axis = np.array([1, 0, 0])  # Pushing along x-axis
    obj_pos += push_distance * push_axis  # Move object position in the push direction

    # Compute position difference (target push position to end-effector position)
    position_diff = obj_pos - eef_pos
    distance = np.linalg.norm(position_diff)

    # Compute rotation difference (object orientation to end-effector orientation)
    rot_diff = robosuite.utils.transform_utils.quat_multiply(obj_quat, robosuite.utils.transform_utils.quat_inverse(eef_quat))
    rot_diff_axis_angle = robosuite.utils.transform_utils.quat2axisangle(rot_diff)

    # Set control gains
    threshold = 0.02  # Distance threshold to stop moving
    Kp_pos = 1  # Position proportional gain
    Kp_rot = 0.00001 # Rotation proportional gain

    # Initialize action array
    action = np.zeros(12)

    # Phase 1: Push the object by moving along the push axis (x-axis)
    if distance > threshold:
        action[:3] = Kp_pos * position_diff  # Proportional control for position

    # Orientation control (align with object)
    action[3:6] = Kp_rot * rot_diff_axis_angle  # Control for orientation

    # Other controls (gripper, base, torso, wheel)
    action[6:7] = 0  # Gripper remains open (can modify based on task)
    action[7:10] = [0, 0, 0]  # base (if applicable)
    action[10:11] = 0         # torso (if applicable)
    action[11:12] = 0         # wheel (optional, if any)

    return action, org_x


def defrost(obs, obj_positions, obj_quats, has_grasped, 
            moved_to_grasp_pos=False, moved_forward=False, moved_back=False, moved_down=False, 
            placed=False, lifted=False, door_opened=False, door_closed=False, 
            obj_placed=False, released=False, pushed_sideways=False, repositioned=False, moved_back_2=False, org_x=0, org_y=0, org_z=0):
    
    # Extract end-effector position and quaternion
    eef_pos = np.array(obs['robot0_eef_pos'])         # End-effector position
    eef_quat = np.array(obs['robot0_eef_quat'])       # End-effector orientation (quaternion)

    # Define object positions and quaternions for microwave and frozen object
    microwave_pos = np.array(obj_positions["microwave_right_group_door"])
    microwave_quat = np.array(obj_quats["microwave_right_group_door"])
    obj_pos = np.array(obs["obj_pos"])
    obj_quat = np.array(obs["obj_quat"])

    # Initialize action array
    action = np.zeros(12)
    

    flag = 0
    # Phase 1: Open microwave door
    if not door_opened:
        action, has_grasped, moved_to_grasp_pos, moved_forward, moved_back, released, pushed_sideways, repositioned, moved_back_2, org_x, org_y = open(
            obs, has_grasped=has_grasped, moved_to_grasp_pos=moved_to_grasp_pos, 
            moved_forward=moved_forward, moved_back=moved_back, released=released, pushed_sideways=pushed_sideways, repositioned=repositioned, moved_back_2=moved_back_2, org_x=org_x, org_y=org_y, 
            target_pos=microwave_pos, target_quat=microwave_quat
        )
        if moved_back_2:
            door_opened = True
            print("Microwave door opened.")
            has_grasped = False
            moved_to_grasp_pos = False
            org_y = eef_pos[0]
            released = False
            moved_back = False

    # Phase 2.2: Place the frozen object in the microwave if it's picked up
    elif not obj_placed:
        action, has_grasped, moved_to_grasp_pos, moved_down, lifted, placed, moved_back, org_x, org_y, org_z = place(
            obs, has_grasped=has_grasped, moved_to_grasp_pos=moved_to_grasp_pos, 
            moved_down=moved_down, lifted=lifted, placed=placed, moved_back=moved_back, org_x=org_x, org_y=org_y, org_z=org_z, flag =flag
        )
        if placed:
            print("Object placed in microwave.")
            has_grasped = False
            moved_to_grasp_pos = False
            released_grasp = False
            obj_placed = True
            moved_back = False
            moved_forward = False
            released = False
            pushed_sideways = False

    # Phase 4: Close the microwave door after placing the object
    elif placed and not released:
        action, has_grasped, moved_to_grasp_pos, moved_forward, pushed_sideways, released, moved_back, org_x, org_y = close(
            obs, has_grasped=has_grasped, moved_to_grasp_pos=moved_to_grasp_pos, moved_forward=moved_forward,
            pushed_forward=pushed_sideways, released_grasp=released, org_x=org_x, org_y=org_y, moved_back=moved_back, target_pos=microwave_pos, target_quat=microwave_quat
        )
        if released:
            print("Microwave door closed.")

    return action, has_grasped, moved_to_grasp_pos, moved_forward, moved_back, moved_down, placed, lifted, door_opened, door_closed, obj_placed, released, pushed_sideways, repositioned, moved_back_2, org_x, org_y, org_z

def placeinoven(obs, obj_positions, obj_quats, has_grasped, 
            moved_to_grasp_pos=False, moved_forward=False, moved_back=False, moved_down=False, 
            placed=False, lifted=False, door_opened=False, door_closed=False, 
            obj_placed=False, released=False, pushed_sideways=False, repositioned=False, moved_back_2=False, cumulative_rot = 0, org_x=0, org_y=0, org_z=0):

    # Extract end-effector position and quaternion
    eef_pos = np.array(obs['robot0_eef_pos'])         # End-effector position
    eef_quat = np.array(obs['robot0_eef_quat'])       # End-effector orientation (quaternion)

    # Define object positions and quaternions for microwave and frozen object
    obj_pos = np.array(obs["obj_pos"])
    obj_quat = np.array(obs["obj_quat"])

    flag = 1

    action = np.zeros(12)

    oven_pos = obj_positions["stovetop_main_group_knob_rear_right"]
    oven_quat = obj_quats["stovetop_main_group_knob_rear_right"]

    if not placed:
        action, has_grasped, moved_to_grasp_pos, moved_down, lifted, placed, moved_back, org_x, org_y, org_z = place(obs=obs, has_grasped=has_grasped, moved_to_grasp_pos=moved_to_grasp_pos, moved_down=moved_down, lifted=lifted, placed=placed, moved_back=moved_back, org_x=org_x, org_y=org_y, org_z=org_z, flag=flag)
        if placed:
            has_grasped = False
            moved_to_grasp_pos = False
            moved_down = False
            lifted = False
            moved_back = False
            placed = True


    # Phase 1: Open microwave door
    if placed and not released:
        print("oven door")
        action, has_grasped, moved_to_grasp_pos, moved_forward, moved_back, released, pushed_sideways, repositioned, moved_back_2, cumulative_rot, org_x, org_y = openoven(
            obs, has_grasped=has_grasped, moved_to_grasp_pos=moved_to_grasp_pos, 
            moved_forward=moved_forward, moved_back=moved_back, released=released, pushed_sideways=pushed_sideways, repositioned=repositioned, moved_back_2=moved_back_2, cumulative_rot=cumulative_rot, org_x=org_x, org_y=org_y, 
            target_pos=oven_pos, target_quat=oven_quat
        )
        if released:
            released = True


    return action, has_grasped, moved_to_grasp_pos, moved_forward, moved_back, moved_down, placed, lifted, door_opened, door_closed, obj_placed, released, pushed_sideways, repositioned, moved_back_2, cumulative_rot, org_x, org_y, org_z


# oven_right_group_main








