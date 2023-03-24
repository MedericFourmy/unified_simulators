import time
import numpy as np
import pinocchio as pin
from example_robot_data import load

def freezed_robot(robot, fixed_joints):
    
    # Remove some joints from pinocchio model
    fixed_ids = [robot.model.getJointId(jname) for jname in fixed_joints] \
                if fixed_joints is not None else []
    # Ugly code to resize model and q0
    rmodel, [gmodel_col, gmodel_vis] = pin.buildReducedModel(
            robot.model, [robot.collision_model, robot.visual_model],
            fixed_ids, robot.q0,
        )
    robot = pin.RobotWrapper(rmodel, gmodel_col, gmodel_vis)
    robot.q0 = robot.model.referenceConfigurations['default']

    return robot


def test_run_simulator(Simulator, robot_name, fixed_joints):

    dur_sim = 20.0

    noise_tau_scale = 0.0

    # force disturbance
    t1_fext, t2_fext = 2.0, 3.0
    # fext = np.array([0, 100, 0, 0, 0, 0])
    # fext = np.array([0, 0, 0, 0, 0, 30])
    fext = np.array([0, 0, 0, 0, 20, 0])
    frame_name = 'panda_link5'

    # Gains are tuned at max before instability for each dt and controller type
    # dt_sim = 1./240
    # # JSID OK
    # Kp = 200
    # Kd = 2
    # # PD+ OK
    # # Kp = 2
    # # Kd = 2*np.sqrt(Kp)

    dt_sim = 1./1000
    # JSID OK
    Kp = 100
    Kd = 2*np.sqrt(Kp)
    # PD+ OK
    # Kp = 40
    # Kd = 2*np.sqrt(Kp)

    N_sim = int(dur_sim/dt_sim)

    robot = load('panda')
    fixed_ids = [robot.model.getJointId(jname) for jname in fixed_joints] \
                if fixed_joints is not None else []

    # Ugly code to resize model and q0
    rmodel, [gmodel_col, gmodel_vis] = pin.buildReducedModel(
            robot.model, [robot.collision_model, robot.visual_model],
            fixed_ids, robot.q0,
        )
    robot = pin.RobotWrapper(rmodel, gmodel_col, gmodel_vis)
    robot.q0 = robot.model.referenceConfigurations['default']
    v0 = np.zeros(robot.model.nv)

    
    sim = Simulator()
    sim.init(dt_sim, robot_name, fixed_joints, visual=True)
    
    q_init = robot.q0.copy() + 0.4  # little offset from stable position
    v_init = v0.copy() + 0.5
    x_init = np.concatenate([q_init, v_init]) 
    sim.setState(x_init)

    print('\n==========================')
    print('Begin simulation + control')
    print(f'Length: {dur_sim} seconds')
    print(f'dt_sim: {1000*dt_sim} milliseconds')
    print('Apply force between ', t1_fext, t2_fext, ' seconds')
    print('   -> ', t1_fext/dt_sim, t2_fext/dt_sim, ' iterations')
    print('fext: ', fext)

    for i in range(N_sim):
        ts = i*dt_sim
        t1 = time.time()
        x = sim.getState()
        q, v = x[:rmodel.nq], x[robot.model.nq:]

        # # ########################
        # # tau_ff = pin.computeGeneralizedGravity(robot.model, robot.data, q)
        # tau_ff = pin.rnea(robot.model, robot.data, q, v, np.zeros(robot.nv))
        
        # # Pure feedforward
        # # tau = tau_ff
        # # PD
        # # tau = - Kp*(q - conf.q0) - Kd*(v - v0)
        # # PD+
        # tau = tau_ff - Kp*(q - conf.q0) - Kd*(v - v0)
        # # ########################

        #########################
        # Joint Space Inverse Dynamics
        ddqd = - Kp*(q - robot.q0) - Kd*(v - v0)
        # M*ddq + C(q)dq + g(q) = tau
        tau = pin.rnea(robot.model, robot.data, q, v, ddqd)
        #########################
        # print(tau)
        # [ 0.    -3.988 -0.644 22.021  0.634  2.278  0.   ]
        # tau [-0.036 -5.268 -0.722 24.31   0.681  2.419  0.009]  # tsid
        # [-7.33564119e-02 -1.65509051e+01 -9.86897279e-01  4.97563905e+01  1.07325654e+00  5.49826615e+00  2.55860554e-03]


        tau_noise = noise_tau_scale*(np.random.random(robot.nv) - 0.5)
        tau += tau_noise

        if t1_fext < ts < t2_fext:
            sim.applyExternalForce(
                fext, frame_name, rf_frame=pin.LOCAL_WORLD_ALIGNED)

        sim.step(tau)

        delay = time.time() - t1
        if delay < dt_sim:
            # print(delay)
            time.sleep(dt_sim - delay)

    print('End')
