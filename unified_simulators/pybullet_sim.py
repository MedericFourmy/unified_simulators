"""
Wrapper class for load a URDF model in both Pinocchio and Bullet.

Example of use:

models = SimuProxy()
models.loadRobotFromErd('talos')
models.loadBulletModel()
models.freeze([
    "arm_left_7_joint",
    "arm_right_7_joint",
    "gripper_left_joint",
    "gripper_right_joint",
    "head_1_joint",
    "head_2_joint",
])
models.setTorqueControlMode()
models.setTalosDefaultFriction()

for i in range(100):
  x = models.getState()
  tau = mycontroller(x)
  models.step(tau)
"""

import pinocchio as pin
import pybullet as pyb
import pybullet_data
import numpy as np
import example_robot_data as robex


class PybulletSim:
    def __init__(self):
        self.readyForSimu = False

        # state variables
        pass
    
    def init(self, dt_sim, robot_name, fixed_joint_names=None, visual=True):
        self.loadRobotFromErd(robot_name)
        pyb_mode = pyb.GUI if visual else pyb.DIRECT
        self.loadBulletModel(dt_sim, guiOpt=pyb_mode)
        if fixed_joint_names is not None:
            self.freeze(fixed_joint_names)
        
        self.setTorqueControlMode()

    def loadRobotFromPath(self, urdf, srdf):
        # TODO
        pass

    def loadRobotFromErd(self, name):
        inst = robex.ROBOTS[name]()

        self.robot = inst.robot

        self.urdf_path = inst.df_path
        self.srdf_path = inst.srdf_path
        self.free_flyer = inst.free_flyer
        self.ref_posture = inst.ref_posture

        # TODO: Useful?
        # pin.loadRotorParameters(self.robot.model, self.srdf_path, False)
        # pin.loadReferenceConfigurations(self.robot.model, self.srdf_path, False)

        self.setPinocchioFinalizationTricks()

    def setPinocchioFinalizationTricks(self):
        # Add free flyers joint limits ... WHY?
        if self.free_flyer:
            self.robot.model.upperPositionLimit[:7] = 1
            self.robot.model.lowerPositionLimit[:7] = -1

        self.robot.model.armature = (
            self.robot.model.rotorInertia * self.robot.model.rotorGearRatio**2
        )
        self.robot.q0 = self.robot.model.referenceConfigurations[self.ref_posture]

        # state variables for external force torques 
        # TODO: use pybullet API instead
        self.tau_fext = np.zeros(self.robot.model.nv)

    ################################################################################
    ################################################################################
    # Load bullet model
    def loadBulletModel(self, dt_sim, guiOpt=pyb.DIRECT):

        self.bulletClient = pyb.connect(guiOpt)

        pyb.setTimeStep(dt_sim)

        # Set gravity (disabled by default in Bullet)
        pyb.setGravity(*(self.robot.model.gravity.linear))

        # Load horizontal plane
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = pyb.loadURDF("plane.urdf")

        if self.free_flyer:
            pose_root = self.robot.q0[:7]
        else:
            pose_root = [0,0,0, 0,0,0,1]
            
        self.robotId = pyb.loadURDF(
            self.urdf_path,
            pose_root[:3],
            pose_root[3:7],
            useFixedBase=not self.free_flyer,
        )

        if self.free_flyer:
            # Magic translation from bullet where the basis center is shifted
            self.localInertiaPos = pyb.getDynamicsInfo(self.robotId, -1)[3]

        self.setBulletFinalizationTrics()

        nq_minus_nqa = 7 if self.free_flyer else 0 
        for ipin, ibul in enumerate(self.bulletCtrlJointsInPinOrder):
            pyb.enableJointForceTorqueSensor(1, ipin, True)
            pyb.resetJointState(self.robotId, ibul, self.robot.q0[nq_minus_nqa + ipin])

    def setBulletFinalizationTrics(self):
        self.bullet_names2indices = {
            pyb.getJointInfo(1, i)[1].decode(): i for i in range(pyb.getNumJoints(self.robotId))
        }
        """
        For free-flyer robots:
        >>> print(list(r.model.names))
        ['universe',
        'root_joint',
        ...]

        For fixed base robots:
        >>> print(list(r.model.names))
        ['universe',
        ...]
        """
        actuated_joint_index_start = 1 + int(self.free_flyer)
        self.bulletCtrlJointsInPinOrder = np.array([
            self.bullet_names2indices[n] for n in self.robot.model.names[actuated_joint_index_start:]
        ])

    ################################################################################

    def freeze(self, fixed_joint_names):
        """
        TODO: not tested for free flyers
        """

        fixed_jids = np.array([self.robot.model.getJointId(jname) for jname in fixed_joint_names])
        fixed_jids_min_universe = fixed_jids - 1
        q0_fixed = self.robot.q0[fixed_jids_min_universe]
        nq_minus_nqa = 7 if self.free_flyer else 0 
        fixed_jids_pyb = self.bulletCtrlJointsInPinOrder[nq_minus_nqa+fixed_jids_min_universe]

        # Build a new reduced model
        self.rmodel_full = self.robot.model
        rmodel, [gmodel_col, gmodel_vis] = pin.buildReducedModel(
            self.rmodel_full, [self.robot.collision_model, self.robot.visual_model],
            fixed_jids.tolist(), self.robot.q0,
        )
        self.robot = pin.RobotWrapper(rmodel, gmodel_col, gmodel_vis)

        # Activate position control on the fixed joints so that they stay put in pybullet
        pyb.setJointMotorControlArray(
            self.robotId,
            jointIndices=fixed_jids_pyb,
            controlMode=pyb.POSITION_CONTROL,
            targetPositions=q0_fixed,
        )

        self.setPinocchioFinalizationTricks()
        self.setBulletFinalizationTrics()

    def setTorqueControlMode(self):
        """
        Disable default position controller in torque controlled joints
        Default controller will take care of other joints
        """
        pyb.setJointMotorControlArray(
            self.robotId,
            jointIndices=self.bulletCtrlJointsInPinOrder,
            controlMode=pyb.VELOCITY_CONTROL,
            forces=[0.0 for _ in self.bulletCtrlJointsInPinOrder],
        )
        self.readyForSimu = True


    # def setTalosDefaultFriction(self):
    #     self.changeFriction(["leg_left_6_joint", "leg_right_6_joint"], 100, 30)

    #        self.changeFriction(["leg_left_5_joint", "leg_right_5_joint"], 100, 30)

    def changeFriction(self, names, lateralFriction=100, spinningFriction=30):
        for n in names:
            idx = self.bullet_names2indices[n]
            pyb.changeDynamics(
                self.robotId,
                idx,
                lateralFriction=lateralFriction,
                spinningFriction=spinningFriction,
            )

    def setGravity(self, g):
        self.robot.model.gravity.linear = g
        pyb.setGravity(*g)

    ################################################################################
    # Called in the sim loop
    ################################################################################

    def step(self, torques):
        assert self.readyForSimu

        # Incorporate torques due to external forces
        torques += self.tau_fext
        pyb.setJointMotorControlArray(
            self.robotId,
            self.bulletCtrlJointsInPinOrder,
            controlMode=pyb.TORQUE_CONTROL,
            forces=torques,
        )
        pyb.stepSimulation()

        # reset external force automatically after each simulation step
        self.tau_fext = np.zeros(self.robot.model.nv)

    def getState(self):
        # Get articulated joint pos and vel
        xbullet = pyb.getJointStates(self.robotId, self.bulletCtrlJointsInPinOrder)
        q = [x[0] for x in xbullet]
        vq = [x[1] for x in xbullet]

        if self.free_flyer:
            # TODO: not much tested
            # Get basis pose
            p, quat = pyb.getBasePositionAndOrientation(self.robotId)
            # Get basis vel
            v, w = pyb.getBaseVelocity(self.robotId)

            # Concatenate into a single x vector
            x = np.concatenate([p, quat, q, v, w, vq])

            # Magic transformation of the basis translation, as classical in Bullet.
            x[:3] -= self.localInertiaPos

        else:
            x = np.concatenate([q, vq])
        return x

    def setState(self, x):
        """Set the robot to the desired states.
        Args:
            q (ndarray): Desired generalized positions.
            dq (ndarray): Desired generalized velocities.
        """
        q, v = x[:self.robot.model.nq], x[self.robot.model.nq:]

        nq_minus_nqa = 7 if self.free_flyer else 0 
        for ipin, ibul in enumerate(self.bulletCtrlJointsInPinOrder):
            pyb.resetJointState(self.robotId, ibul, q[nq_minus_nqa + ipin], v[nq_minus_nqa + ipin])

    def applyExternalForce(self, f, ee_frame, rf_frame=pin.LOCAL_WORLD_ALIGNED):
        # Store the torque due to exterior forces for simulation step

        x = self.getState()
        q = x[:self.robot.model.nq]
        self.robot.framesForwardKinematics(q)
        self.robot.computeJointJacobians(q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        Jf = pin.getFrameJacobian(
            self.robot.model, self.robot.data, self.robot.model.getFrameId(ee_frame), rf_frame)
        
        # The torques from different forces accumulate
        self.tau_fext += Jf.T @ f


if __name__ == "__main__":
    dt_sim = 1e-3
    robot_name = 'panda'
    fixed_indexes = []
    # fixed_indexes = [2,5,6,7]
    fixed_joints =  [f'panda_joint{i}' for i in fixed_indexes] + ['panda_finger_joint1', 'panda_finger_joint2']
    # fixed_joints = None

    from utils import test_run_simulator
    np.set_printoptions(linewidth=150)
    test_run_simulator(PybulletSim, robot_name, fixed_joints)