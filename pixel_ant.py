# Copyright 2019 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Quadruped Domain."""
import cv2
import collections, time
import numpy as np

from dm_control import suite
from dm_control import mujoco
from dm_control.mujoco.wrapper import mjbindings
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import xml_tools
from lxml import etree
from scipy import ndimage
from gymnasium.spaces import Box
from gymnasium.core import Env
from collections import deque

enums = mjbindings.enums
mjlib = mjbindings.mjlib


_DEFAULT_TIME_LIMIT = 20
_CONTROL_TIMESTEP = .02

# Horizontal speeds above which the move reward is 1.
_RUN_SPEED = 5
_WALK_SPEED = 0.5

# Constants related to terrain generation.
_HEIGHTFIELD_ID = 0
_TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
_TERRAIN_BUMP_SCALE = 2  # Spatial scale of terrain bumps (in meters).

# Named model elements.
_TOES = ['toe_front_left', 'toe_back_left', 'toe_back_right', 'toe_front_right']
_WALLS = ['wall_px', 'wall_py', 'wall_nx', 'wall_ny']

SUITE = containers.TaggedTasks()


def make_model(floor_size=None, terrain=False, rangefinders=False,
               walls=True, target=False, ball=False):
  """Returns the model XML string."""
  xml_string = common.read_model('quadruped.xml')
  parser = etree.XMLParser(remove_blank_text=True)
  mjcf = etree.XML(xml_string, parser)

  # Set floor size.
  if floor_size is not None:
    floor_geom = mjcf.find('.//geom[@name=\'floor\']')
    floor_geom.attrib['size'] = f'{floor_size} {floor_size} .5'
    
    wall_px_geom = mjcf.find('.//geom[@name=\'wall_px\']')
    wall_px_geom.attrib['pos'] = f'{-floor_size-0.7} 0 .7'
    wall_px_geom.attrib['size'] = f'1 {floor_size} .5'

    wall_py_geom = mjcf.find('.//geom[@name=\'wall_py\']')
    wall_py_geom.attrib['pos'] = f'0 {-floor_size-0.7} .7'
    wall_py_geom.attrib['size'] = f'{floor_size} 1 .5'

    wall_nx_geom = mjcf.find('.//geom[@name=\'wall_nx\']')
    wall_nx_geom.attrib['pos'] = f'{floor_size+0.7} 0 .7'
    wall_nx_geom.attrib['size'] = f'1 {floor_size} .5'

    wall_ny_geom = mjcf.find('.//geom[@name=\'wall_ny\']')
    wall_ny_geom.attrib['pos'] = f'0 {floor_size+0.7} .7'
    wall_ny_geom.attrib['size'] = f'{floor_size} 1 .5'

    camera_geom = mjcf.find('.//camera[@name=\'global\']')
    camera_geom.attrib['pos'] = f'{-floor_size} {floor_size} {floor_size}'

  # Remove walls, ball and target.
  if not walls:
    for wall in _WALLS:
      wall_geom = xml_tools.find_element(mjcf, 'geom', wall)
      wall_geom.getparent().remove(wall_geom)

  if not ball:
    # Remove ball.
    ball_body = xml_tools.find_element(mjcf, 'body', 'ball')
    ball_body.getparent().remove(ball_body)

  if not target:
    # Remove target.
    target_site = xml_tools.find_element(mjcf, 'site', 'target')
    target_site.getparent().remove(target_site)

  # Remove terrain.
  if not terrain:
    terrain_geom = xml_tools.find_element(mjcf, 'geom', 'terrain')
    terrain_geom.getparent().remove(terrain_geom)

  # Remove rangefinders if they're not used, as range computations can be
  # expensive, especially in a scene with heightfields.
  if not rangefinders:
    rangefinder_sensors = mjcf.findall('.//rangefinder')
    for rf in rangefinder_sensors:
      rf.getparent().remove(rf)

  return etree.tostring(mjcf, pretty_print=True)

@SUITE.add()
def reach_target(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the reach task."""
  xml_string = make_model(walls=True, floor_size=8, ball=False, target=True)
  physics = Physics.from_xml_string(xml_string, common.ASSETS)
  task = Reach(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             control_timestep=_CONTROL_TIMESTEP,
                             **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Quadruped domain."""

  def _reload_from_data(self, data):
    super()._reload_from_data(data)
    # Clear cached sensor names when the physics is reloaded.
    self._sensor_types_to_names = {}
    self._hinge_names = []

  def _get_sensor_names(self, *sensor_types):
    try:
      sensor_names = self._sensor_types_to_names[sensor_types]
    except KeyError:
      [sensor_ids] = np.where(np.in1d(self.model.sensor_type, sensor_types))
      sensor_names = [self.model.id2name(s_id, 'sensor') for s_id in sensor_ids]
      self._sensor_types_to_names[sensor_types] = sensor_names
    return sensor_names

  def torso_upright(self):
    """Returns the dot-product of the torso z-axis and the global z-axis."""
    return np.asarray(self.named.data.xmat['torso', 'zz'])

  def torso_velocity(self):
    """Returns the velocity of the torso, in the local frame."""
    return self.named.data.sensordata['velocimeter'].copy()

  def egocentric_state(self):
    """Returns the state without global orientation or position."""
    if not self._hinge_names:
      [hinge_ids] = np.nonzero(self.model.jnt_type ==
                               enums.mjtJoint.mjJNT_HINGE)
      self._hinge_names = [self.model.id2name(j_id, 'joint')
                           for j_id in hinge_ids]
    return np.hstack((self.named.data.qpos[self._hinge_names],
                      self.named.data.qvel[self._hinge_names],
                      self.data.act))

  def toe_positions(self):
    """Returns toe positions in egocentric frame."""
    torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
    torso_pos = self.named.data.xpos['torso']
    torso_to_toe = self.named.data.xpos[_TOES] - torso_pos
    return torso_to_toe.dot(torso_frame)

  def force_torque(self):
    """Returns scaled force/torque sensor readings at the toes."""
    force_torque_sensors = self._get_sensor_names(enums.mjtSensor.mjSENS_FORCE,
                                                  enums.mjtSensor.mjSENS_TORQUE)
    return np.arcsinh(self.named.data.sensordata[force_torque_sensors])

  def imu(self):
    """Returns IMU-like sensor readings."""
    imu_sensors = self._get_sensor_names(enums.mjtSensor.mjSENS_GYRO,
                                         enums.mjtSensor.mjSENS_ACCELEROMETER)
    return self.named.data.sensordata[imu_sensors]

  def rangefinder(self):
    """Returns scaled rangefinder sensor readings."""
    rf_sensors = self._get_sensor_names(enums.mjtSensor.mjSENS_RANGEFINDER)
    rf_readings = self.named.data.sensordata[rf_sensors]
    no_intersection = -1.0
    return np.where(rf_readings == no_intersection, 1.0, np.tanh(rf_readings))

  def origin_distance(self):
    """Returns the distance from the origin to the workspace."""
    return np.asarray(np.linalg.norm(self.named.data.site_xpos['workspace']))

  def origin(self):
    """Returns origin position in the torso frame."""
    torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
    torso_pos = self.named.data.xpos['torso']
    return -torso_pos.dot(torso_frame)

  def ball_state(self):
    """Returns ball position and velocity relative to the torso frame."""
    data = self.named.data
    torso_frame = data.xmat['torso'].reshape(3, 3)
    ball_rel_pos = data.xpos['ball'] - data.xpos['torso']
    ball_rel_vel = data.qvel['ball_root'][:3] - data.qvel['root'][:3]
    ball_rot_vel = data.qvel['ball_root'][3:]
    ball_state = np.vstack((ball_rel_pos, ball_rel_vel, ball_rot_vel))
    return ball_state.dot(torso_frame).ravel()

  def target_position(self):
    """Returns target position in torso frame."""
    torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
    torso_pos = self.named.data.xpos['torso']
    torso_to_target = self.named.data.site_xpos['target'] - torso_pos
    return torso_to_target.dot(torso_frame)

  def ball_to_target_distance(self):
    """Returns horizontal distance from the ball to the target."""
    ball_to_target = (self.named.data.site_xpos['target'] -
                      self.named.data.xpos['ball'])
    return np.linalg.norm(ball_to_target[:2])

  def self_to_ball_distance(self):
    """Returns horizontal distance from the quadruped workspace to the ball."""
    self_to_ball = (self.named.data.site_xpos['workspace']
                    -self.named.data.xpos['ball'])
    return np.linalg.norm(self_to_ball[:2])

  def self_to_target_distance(self):
    """Returns horizontal distance from the quadruped workspace to the target."""
    self_to_target = (self.named.data.site_xpos['workspace']
                    -self.named.data.site_xpos['target'])
    return np.linalg.norm(self_to_target[:2])


def _find_non_contacting_height(physics, orientation, x_pos=0.0, y_pos=0.0):
  """Find a height with no contacts given a body orientation.

  Args:
    physics: An instance of `Physics`.
    orientation: A quaternion.
    x_pos: A float. Position along global x-axis.
    y_pos: A float. Position along global y-axis.
  Raises:
    RuntimeError: If a non-contacting configuration has not been found after
    10,000 attempts.
  """
  z_pos = 0.0  # Start embedded in the floor.
  num_contacts = 1
  num_attempts = 0
  # Move up in 1cm increments until no contacts.
  while num_contacts > 0:
    try:
      with physics.reset_context():
        physics.named.data.qpos['root'][:3] = x_pos, y_pos, z_pos
        physics.named.data.qpos['root'][3:] = orientation
    except control.PhysicsError:
      # We may encounter a PhysicsError here due to filling the contact
      # buffer, in which case we simply increment the height and continue.
      pass
    num_contacts = physics.data.ncon
    z_pos += 0.01
    num_attempts += 1
    if num_attempts > 10000:
      raise RuntimeError('Failed to find a non-contacting configuration.')


def _common_observations(physics):
  """Returns the observations common to all tasks."""
  obs = collections.OrderedDict()
  obs['egocentric_state'] = physics.egocentric_state()
  obs['torso_velocity'] = physics.torso_velocity()
  obs['torso_upright'] = physics.torso_upright()
  obs['imu'] = physics.imu()
  obs['force_torque'] = physics.force_torque()
  return obs


def _upright_reward(physics, deviation_angle=0):
  """Returns a reward proportional to how upright the torso is.

  Args:
    physics: an instance of `Physics`.
    deviation_angle: A float, in degrees. The reward is 0 when the torso is
      exactly upside-down and 1 when the torso's z-axis is less than
      `deviation_angle` away from the global z-axis.
  """
  deviation = np.cos(np.deg2rad(deviation_angle))
  return rewards.tolerance(
      physics.torso_upright(),
      bounds=(deviation, float('inf')),
      sigmoid='linear',
      margin=1 + deviation,
      value_at_margin=0)


class Reach(base.Task):
  """A quadruped task solved by bringing a ball to the origin."""

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    Args:
      physics: An instance of `Physics`.

    """
    # Initial configuration, random azimuth and horizontal position.
    azimuth = self.random.uniform(0, 2*np.pi)
    orientation = np.array((np.cos(azimuth/2), 0, 0, np.sin(azimuth/2)))
    spawn_radius = 0.9 * physics.named.model.geom_size['floor', 0]
    x_pos, y_pos = self.random.uniform(-spawn_radius, spawn_radius, size=(2,))
    _find_non_contacting_height(physics, orientation, x_pos, y_pos)

    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation to the agent."""
    obs = _common_observations(physics)
    obs['target_position'] = physics.target_position()
    # obs['ball_state'] = physics.ball_state()
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""

    # Reward for moving close to the ball.
    arena_radius = physics.named.model.geom_size['floor', 0] * np.sqrt(2)
    workspace_radius = physics.named.model.site_size['workspace', 0]
    reach_reward = rewards.tolerance(
        physics.self_to_target_distance(),
        bounds=(0, workspace_radius),
        sigmoid='linear',
        margin=arena_radius, value_at_margin=0) 

    return _upright_reward(physics) * reach_reward


class Observation:
  """ Dummy class """
  def __init__(self, image, proprioception):
    self.image = image
    self.proprioception = proprioception


class VisualAntReacher(Env):
  def __init__(self, **kwargs):
    self.env = reach_target()
    self.rgb_array = kwargs.get('render_mode', '') == "rgb_array"
    self.proprioception_keys = ['egocentric_state', 'torso_velocity', 'torso_upright', 'target_position']

    # Observation space
    self._obs_dim = 0
    for key, val in self.env.observation_spec().items():
      if key in self.proprioception_keys:
        if val.shape:
          self._obs_dim += val.shape[0]
        else:
          self._obs_dim += 1

    # Action space
    self._action_dim = self.env.action_spec().shape[0]
    self._use_image = True
    self._image_history_len = 3
    self.stacked_frames = deque(maxlen=self._image_history_len)

  def get_observation(self, time_step):
    """_summary_

    Args:
        time_step (_type_): _description_

    Returns:
        _type_: _description_
    """
    proprioception = []
    for key in self.proprioception_keys:
      proprioception.append(time_step.observation[key].ravel())
    proprioception = np.concatenate(proprioception)

    # Render the environment from multiple camera views
    camera_ids = [0, 3] # Get overhead camera and ego-centric camera
    # camera_ids = [0]      # Get overhead camera only
    frames = []
    for camera_id in camera_ids:
        pixels = self.env.physics.render(camera_id=camera_id, width=84, height=84)
        # Convert RGB to BGR for OpenCV
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        frames.append(pixels)

    image = np.concatenate(frames, axis=-1)
    image = np.transpose(image, (2, 0, 1))
    return image, proprioception
  
  def reset(self, *, seed = None, options = None):
    time_step = self.env.reset()
    info = {}
    image, proprioception = self.get_observation(time_step)
    for _ in range(self._image_history_len):
      self.stacked_frames.append(image)
    obs = Observation(np.concatenate(self.stacked_frames), proprioception)
    return obs, info
  
  def step(self, action):
    time_step = self.env.step(action)
    reward = time_step.reward
    terminated = time_step.last()
    truncated = False
    info = {}
    image, proprioception = self.get_observation(time_step)
    self.stacked_frames.append(image)
    obs = Observation(np.concatenate(self.stacked_frames), proprioception)
    return obs, reward, terminated, truncated, info

  @property
  def observation_space(self):
      return Box(shape=(self._obs_dim,), high=10, low=-10)

  @property
  def image_space(self):
      image_shape = (6 * self.stacked_frames.maxlen, 84, 84)
      return Box(low=0, high=255, shape=image_shape)

  @property
  def proprioception_space(self):
      return self.observation_space

  @property
  def action_space(self):
      return Box(shape=(self._action_dim,), high=1, low=-1)

  def render(self):
      if self.rgb_array:
          rgb_array = self.env.physics.render(camera_id=0)
          return rgb_array
      
      self.env.render()  


def simple_env():
    # Load the Reach environment
    env = reach_target()

    # Run the environment loop
    time_step = env.reset()
    print(time_step.observation.keys())
    while not time_step.last():
        # Sample a random action
        action = np.random.uniform(env.action_spec().minimum,
                                   env.action_spec().maximum,
                                   size=env.action_spec().shape)
        time_step = env.step(action)
        print(time_step.observation.keys())
        # print(f"Reward: {time_step.reward}, Observation: {time_step.observation['egocentric_state']}")

        # Render the environment from multiple camera views
        camera_ids = [0, 1, 2, 3]
        frames = []
        for camera_id in camera_ids:
            pixels = env.physics.render(camera_id=camera_id, width=84, height=84)
            # Convert RGB to BGR for OpenCV
            pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            frames.append(pixels)
        
        # Concatenate frames horizontally
        combined_frame = cv2.hconcat(frames)
        cv2.imshow('Environment', combined_frame)
        time.sleep(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
  # simple_env()
  env = VisualAntReacher()

  for EP in range(10):
    obs, _ = env.reset()
    terminated, truncated = False, False
    ret, steps = 0, 0
    while not (terminated or truncated):
      # Concatenate frames horizontally
      cv2.imshow('Environment', np.transpose(obs.image, (1, 2, 0))[:, :, -3:])
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
        
      action = env.action_space.sample()
      obs, reward, terminated, truncated, info = env.step(action)
      next_obs = obs
      ret += reward
      steps += 1

    print(f"Episode {EP} ended with return {ret:.2f} in {steps} timesteps.")