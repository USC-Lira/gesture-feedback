from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS, MULTI_STAGE_TASK_DATASETS
from robocasa.utils.dataset_registry import get_ds_path
from robocasa.utils.env_utils import create_env, run_random_rollouts

import numpy as np

"""
Select a random task (that comes with an accompanying dataset) to run rollouts for.
Alternatively, sample *any* kitchen task in RoboCasa by replacing the following line with
env_name = np.random.choice(list(ALL_KITCHEN_ENVIRONMENTS))
"""
# env_name = np.random.choice(
#     list(SINGLE_STAGE_TASK_DATASETS)
# )
print(list(MULTI_STAGE_TASK_DATASETS))
env_name = "MicrowaveThawing"

"""
['PnPCounterToCab', 'PnPCabToCounter', 'PnPCounterToSink', 'PnPSinkToCounter', 'PnPCounterToMicrowave', 
'PnPMicrowaveToCounter', 'PnPCounterToStove', 'PnPStoveToCounter', 'OpenSingleDoor', 'CloseSingleDoor', 
'OpenDoubleDoor', 'CloseDoubleDoor', 'OpenDrawer', 'CloseDrawer', 'TurnOnSinkFaucet', 'TurnOffSinkFaucet', 
'TurnSinkSpout', 'TurnOnStove', 'TurnOffStove', 'CoffeeSetupMug', 'CoffeeServeMug', 'CoffeePressButton', 
'TurnOnMicrowave', 'TurnOffMicrowave', 'NavigateKitchen']
"""

"""
['ArrangeVegetables', 'MicrowaveThawing', 'RestockPantry', 'PreSoakPan', 'PrepareCoffee']
"""

print(env_name)

# seed environment as needed. set seed=None to run unseeded
env = create_env(env_name=env_name, seed=0)

# run rollouts with random actions and save video
info = run_random_rollouts(
    env, num_rollouts=1, num_steps=1100, video_path="/tmp/test.mp4"
)
print(info)