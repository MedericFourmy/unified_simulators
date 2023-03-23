import numpy as np
import pinocchio as pin
from functools import partial
import example_robot_data as robex


class PinocchioSim:
    def __init__(self):
        self.readyForSimu = False
        pass

    def init(self, dt_sim, robot_name, fixed_joint_names=None, visual=True):
        self.loadRobotFromErd(robot_name)
        if fixed_joint_names is not None:
            self.freeze(fixed_joint_names)
        
        self.dt_sim = dt_sim
        self.visual = visual
        if visual:
            self.robot.initViewer(loadModel=True)

        self.q = np.zeros(self.robot.nq)
        self.v = np.zeros(self.robot.nv)
        self.dv = np.zeros(self.robot.nv)
        self.tau_fext = np.zeros(self.robot.nv)

    def loadRobotFromPath(self, urdf, srdf):
        # TODO
        pass

    def loadRobotFromErd(self, name):
        inst = robex.ROBOTS[name]()

        self.robot = inst.robot
        self.rmodel = inst.robot.model
        self.gmodel_col = inst.robot.collision_model
        self.gmodel_vis = inst.robot.visual_model

        self.urdf_path = inst.df_path
        self.srdf_path = inst.srdf_path
        self.free_flyer = inst.free_flyer
        self.ref_posture = inst.ref_posture

        self.setPinocchioFinalizationTricks()

    def setPinocchioFinalizationTricks(self):
        # Add free flyers joint limits ... WHY?
        if self.free_flyer:
            self.rmodel.upperPositionLimit[:7] = 1
            self.rmodel.lowerPositionLimit[:7] = -1

        self.rmodel.armature = (
            self.rmodel.rotorInertia * self.rmodel.rotorGearRatio**2
        )
        self.rmodel.q0 = self.rmodel.referenceConfigurations[self.ref_posture]
        self.rdata = self.rmodel.createData()

    def freeze(self, jointNames):

        jointIds = [self.rmodel.getJointId(jname) for jname in jointNames]

        self.rmodel_full = self.rmodel
        self.rmodel, [self.gmodel_col, self.gmodel_vis] = pin.buildReducedModel(
            self.rmodel_full,
            [self.gmodel_col, self.gmodel_vis],
            jointIds,
            self.rmodel.q0,
        )
        self.robot = pin.RobotWrapper(self.rmodel, self.gmodel_col, self.gmodel_vis)

        self.setPinocchioFinalizationTricks()

    def setGravity(self, g):
        self.robot.model.gravity.linear = g

    ################################################################################
    # Called in the sim loop
    ################################################################################

    def step(self, tau):
        assert tau.shape[0] == self.rmodel.nv
        self.tau_cmd = tau

        # Free foward dynamics algorithm
        tau = self.tau_cmd + self.tau_fext

        # NO ARMATURE
        self.dv = pin.aba(self.robot.model, self.robot.data, self.q, self.v, tau)

        # # WITH ARMATURE
        # h = pin.rnea(self.robot.model, self.robot.data, self.q, self.v, np.zeros(self.robot.nv))
        # M = pin.crba(self.robot.model, self.robot.data, self.q)
        # armature = 0.1*np.ones(7)
        # # M += np.diag(armature)
        # Minv = np.linalg.inv(M)
        # self.dv = Minv @ (tau - h)

        # u,s,vh = np.linalg.svd(M)
        # print(s)
        # print('M condition #: ', s[0]/s[-1])

        # Exact integration for this simple linear system x=(q,v), u=(dv)
        self.q = self.q + self.v*self.dt_sim + 0.5*self.dv*self.dt_sim**2
        self.v = self.v + self.dv*self.dt_sim

        # update visuals
        if self.visual:
            self.robot.display(self.q)

        # reset external force automatically after each simulation step
        self.tau_fext = np.zeros(self.rmodel.nv)

    def getState(self):
        # Make a explicit copies to avoid silly blinders
        return np.concatenate([self.q.copy(), self.v.copy()])

    def setState(self, x):
        # Make a explicit copies to avoid silly blinders
        self.q, self.v = x[:self.robot.model.nq], x[self.robot.model.nq:]

    def applyExternalForce(self, f, ee_frame, rf_frame=pin.LOCAL_WORLD_ALIGNED):
        # Store the torque due to exterior forces for simulation step

        self.robot.framesForwardKinematics(self.q)
        self.robot.computeJointJacobians(self.q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        Jf = pin.getFrameJacobian(
            self.robot.model, self.robot.data, self.robot.model.getFrameId(ee_frame), rf_frame)
        
        # The torques from different forces accumulate
        self.tau_fext += Jf.T @ f


if __name__ == "__main__":
    dt_sim = 1e-3
    robot_name = 'panda'
    fixed_joints = ['panda_finger_joint1', 'panda_finger_joint2']
    # fixed_joints = None

    from utils import test_run_simulator
    np.set_printoptions(linewidth=150)
    test_run_simulator(PinocchioSim, robot_name, fixed_joints)