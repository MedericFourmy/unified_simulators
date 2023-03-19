import numpy as np
import pinocchio as pin
from functools import partial


class PinocchioSim:

    def __init__(self, dt_sim, urdf_path, package_dirs, joint_names=None, base_pose=[0, 0, 0, 0, 0, 0, 1], visual=True):

        self.dt_sim = dt_sim
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs)
        self.visual = visual
        if visual:
            self.robot.initViewer(loadModel=True)

        self.nq = self.robot.nq
        self.nv = self.robot.nv
        self.q = np.zeros(self.robot.nq)
        self.dq = np.zeros(self.robot.nv)
        self.ddq = np.zeros(self.robot.nv)
        self.tau_cmd = np.zeros(self.robot.nv)
        self.tau_fext = np.zeros(self.robot.nv)

        self.pin_integrate = partial(pin.integrate, self.robot.model)

    def get_state(self):
        # Make a explicit copies to avoid silly blinders
        return self.q.copy(), self.dq.copy(), self.ddq.copy()

    def set_state(self, q, dq, ddq=None):
        # Make a explicit copies to avoid silly blinders
        self.q = q.copy()
        self.dq = dq.copy()
        if ddq is None:
            self.ddq = np.zeros(self.nv)

    def apply_external_force(self, f, ee_frame, rf_frame=pin.LOCAL_WORLD_ALIGNED):
        # Store the torque due to exterior forces for simulation step

        self.robot.framesForwardKinematics(self.q)
        self.robot.computeJointJacobians(self.q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        Jf = pin.getFrameJacobian(
            self.robot.model, self.robot.data, self.robot.model.getFrameId(ee_frame), rf_frame)
        self.tau_fext = Jf.T @ f

    def send_joint_command(self, tau):
        """Apply the desired torques to the joints.
        Args:
            tau (ndarray): Torque to be applied.
        """
        assert tau.shape[0] == self.nv
        self.tau_cmd = tau

    def step_simulation(self):
        """Step the simulation forward."""

        # Free foward dynamics algorithm
        tau = self.tau_cmd + self.tau_fext

        # NO ARMATURE
        self.ddq = pin.aba(self.robot.model, self.robot.data,
                           self.q, self.dq, tau)

        # # WITH ARMATURE
        # h = pin.rnea(self.robot.model, self.robot.data, self.q, self.dq, np.zeros(self.robot.nv))
        # M = pin.crba(self.robot.model, self.robot.data, self.q)
        # armature = 0.1*np.ones(7)
        # # M += np.diag(armature)
        # Minv = np.linalg.inv(M)
        # self.ddq = Minv @ (tau - h)

        # u,s,vh = np.linalg.svd(M)
        # print(s)
        # print('M condition #: ', s[0]/s[-1])

        # Exact integration for this simple linear system x=(q,dq), u=(ddq)
        self.q = self.q + self.dq*self.dt_sim + 0.5*self.ddq*self.dt_sim**2
        self.dq = self.dq + self.ddq*self.dt_sim

        # update visuals
        if self.visual:
            self.robot.display(self.q)

        # reset external force automatically after each simulation step
        self.tau_fext = np.zeros(self.nv)



if __name__ == '__main__':
    from utils import test_run_simulator
    import config_panda as conf
    test_run_simulator(PinocchioSim, conf)