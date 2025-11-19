# This software contains source code provided by NVIDIA Corporation.
# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import omni.timeline
import omni.ui as ui
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.stage import create_new_stage, get_current_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.gui.components.element_wrappers import CollapsableFrame, StateButton
from isaacsim.gui.components.element_wrappers.core_connectors import LoadButton, ResetButton
from isaacsim.gui.components.ui_utils import get_style
from omni.usd import StageEventType
from pxr import Sdf, UsdLux

from .scenario import UR10TrajectoryGenerationExample


class UIBuilder:
    def __init__(self):
        # Frames are sub-windows that can contain multiple UI elements
        self.frames = []
        # UI elements created using a UIElementWrapper instance
        self.wrapped_ui_elements = []

        # Get access to the timeline to control stop/pause/play programmatically
        self._timeline = omni.timeline.get_timeline_interface()

        # Run initialization for the provided example
        self._on_init()

    ###################################################################################
    #           The Functions Below Are Called Automatically By extension.py
    ###################################################################################

    def on_menu_callback(self):
        """Callback for when the UI is opened from the toolbar.
        This is called directly after build_ui().
        """
        pass

    def on_timeline_event(self, event):
        """Callback for Timeline events (Play, Pause, Stop)

        Args:
            event (omni.timeline.TimelineEventType): Event Type
        """
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            # When the user hits the stop button through the UI, they will inevitably discover edge cases where things break
            # For complete robustness, the user should resolve those edge cases here
            # In general, for extensions based off this template, there is no value to having the user click the play/stop
            # button instead of using the Load/Reset/Run buttons provided.
            self._cspace_trajectory_btn.reset()
            self._taskspace_trajectory_btn.reset()
            self._advanced_trajectory_btn.reset()
            self._cspace_trajectory_btn.enabled = False
            self._taskspace_trajectory_btn.enabled = False
            self._advanced_trajectory_btn.enabled = False

    def on_physics_step(self, step: float):
        """Callback for Physics Step.
        Physics steps only occur when the timeline is playing

        Args:
            step (float): Size of physics step
        """
        pass

    def on_stage_event(self, event):
        """Callback for Stage Events

        Args:
            event (omni.usd.StageEventType): Event Type
        """
        if event.type == int(StageEventType.OPENED):
            # If the user opens a new stage, the extension should completely reset
            self._reset_extension()

    def cleanup(self):
        """
        Called when the stage is closed or the extension is hot reloaded.
        Perform any necessary cleanup such as removing active callback functions
        Buttons imported from isaacsim.gui.components.element_wrappers implement a cleanup function that should be called
        """
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()

    def build_ui(self):
        """
        Build a custom UI tool to run your extension.
        This function will be called any time the UI window is closed and reopened.
        """
        world_controls_frame = CollapsableFrame("World Controls", collapsed=False)

        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._load_btn = LoadButton(
                    "Load Button", "LOAD", setup_scene_fn=self._setup_scene, setup_post_load_fn=self._setup_scenario
                )
                self._load_btn.set_world_settings(physics_dt=1 / 60.0, rendering_dt=1 / 60.0)
                self.wrapped_ui_elements.append(self._load_btn)

                self._reset_btn = ResetButton(
                    "Reset Button", "RESET", pre_reset_fn=None, post_reset_fn=self._on_post_reset_btn
                )
                self._reset_btn.enabled = False
                self.wrapped_ui_elements.append(self._reset_btn)

        run_scenario_frame = CollapsableFrame("Run Scenario", collapsed=False)

        with run_scenario_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._cspace_trajectory_btn = StateButton(
                    "Run CSpace Trajectory",
                    "CSPACE TRAJECTORY",
                    "STOP",
                    on_a_click_fn=self._on_run_cspace,
                    on_b_click_fn=self._on_stop,
                    physics_callback_fn=self._update_scenario,
                )
                self._cspace_trajectory_btn.enabled = False

                self._taskspace_trajectory_btn = StateButton(
                    "Run TaskSpace Trajectory",
                    "TASKSPACE TRAJECTORY",
                    "STOP",
                    on_a_click_fn=self._on_run_taskspace,
                    on_b_click_fn=self._on_stop,
                    physics_callback_fn=self._update_scenario,
                )
                self._taskspace_trajectory_btn.enabled = False

                self._advanced_trajectory_btn = StateButton(
                    "Run Advanced Trajectory",
                    "ADVANCED TRAJECTORY",
                    "STOP",
                    on_a_click_fn=self._on_run_advanced,
                    on_b_click_fn=self._on_stop,
                    physics_callback_fn=self._update_scenario,
                )
                self._advanced_trajectory_btn.enabled = False

                self.wrapped_ui_elements.append(self._cspace_trajectory_btn)
                self.wrapped_ui_elements.append(self._taskspace_trajectory_btn)
                self.wrapped_ui_elements.append(self._advanced_trajectory_btn)

    ######################################################################################
    # Functions Below This Point Support The Provided Example And Can Be Deleted/Replaced
    ######################################################################################

    def _on_init(self):
        self._articulation = None
        self._cuboid = None
        self._scenario = UR10TrajectoryGenerationExample()

    def _add_light_to_stage(self):
        """
        A new stage does not have a light by default.  This function creates a spherical light
        """
        sphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])

    def _setup_scene(self):
        """
        This function is attached to the Load Button as the setup_scene_fn callback.
        On pressing the Load Button, a new instance of World() is created and then this function is called.
        The user should now load their assets onto the stage and add them to the World Scene.
        """
        create_new_stage()
        self._add_light_to_stage()
        set_camera_view(eye=[2.5, 2, 2.5], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")

        loaded_objects = self._scenario.load_example_assets()

        # Add user-loaded objects to the World
        world = World.instance()
        for loaded_object in loaded_objects:
            world.scene.add(loaded_object)

    def _setup_scenario(self):
        """
        This function is attached to the Load Button as the setup_post_load_fn callback.
        The user may assume that their assets have been loaded by their setup_scene_fn callback, that
        their objects are properly initialized, and that the timeline is paused on timestep 0.

        In this example, a scenario is initialized which will move each robot joint one at a time in a loop while moving the
        provided prim in a circle around the robot.
        """
        self._scenario.setup()

        # UI management
        self._cspace_trajectory_btn.enabled = True
        self._taskspace_trajectory_btn.enabled = True
        self._advanced_trajectory_btn.enabled = True
        self._reset_btn.enabled = True

    def _on_post_reset_btn(self):
        """
        This function is attached to the Reset Button as the post_reset_fn callback.
        The user may assume that their objects are properly initialized, and that the timeline is paused on timestep 0.

        They may also assume that objects that were added to the World.Scene have been moved to their default positions.
        I.e. the cube prim will move back to the position it was in when it was created in self._setup_scene().
        """
        self._scenario.reset()
        self._cspace_trajectory_btn.reset()
        self._taskspace_trajectory_btn.reset()
        self._advanced_trajectory_btn.reset()
        self._cspace_trajectory_btn.enabled = True
        self._taskspace_trajectory_btn.enabled = True
        self._advanced_trajectory_btn.enabled = True

    def _update_scenario(self, step: float):
        """This function is attached to the Run Scenario StateButton.
        This function was passed in as the physics_callback_fn argument.
        This means that when the a_text "RUN" is pressed, a subscription is made to call this function on every physics step.
        When the b_text "STOP" is pressed, the physics callback is removed.

        Args:
            step (float): The dt of the current physics step
        """
        self._scenario.update(step)

    def _on_run_cspace(self):
        self._timeline.play()
        self._scenario.reset()
        self._taskspace_trajectory_btn.reset()
        self._advanced_trajectory_btn.reset()
        self._scenario.setup_cspace_trajectory()

    def _on_run_taskspace(self):
        self._timeline.play()
        self._scenario.reset()
        self._cspace_trajectory_btn.reset()
        self._advanced_trajectory_btn.reset()
        self._scenario.setup_taskspace_trajectory()

    def _on_run_advanced(self):
        self._timeline.play()
        self._scenario.reset()
        self._cspace_trajectory_btn.reset()
        self._taskspace_trajectory_btn.reset()
        self._scenario.setup_advanced_trajectory()

    def _on_stop(self):
        self._timeline.pause()

    def _reset_extension(self):
        """This is called when the user opens a new stage from self.on_stage_event().
        All state should be reset.
        """
        self._on_init()
        self._reset_ui()

    def _reset_ui(self):
        self._cspace_trajectory_btn.reset()
        self._taskspace_trajectory_btn.reset()
        self._advanced_trajectory_btn.reset()
        self._cspace_trajectory_btn.enabled = False
        self._taskspace_trajectory_btn.enabled = False
        self._advanced_trajectory_btn.enabled = False
        self._reset_btn.enabled = False
