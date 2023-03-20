import time
import numpy as np
import pinocchio as pin


def test_run_simulator(Simulator, conf):

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

    # gravity = np.zeros(3)
    # gravity = np.array([0,0,-9.81])
    robot = pin.RobotWrapper.BuildFromURDF(conf.urdf_path, conf.package_dirs)
    # robot.model.gravity.linear = gravity

    sim = Simulator(dt_sim, conf.urdf_path, conf.package_dirs,
                    conf.joint_names, visual=True)
    # sim.set_gravity(gravity)
    # q_init = conf.q0
    q_init = conf.q0 + 0.1  # little offset from stable position
    sim.set_state(q_init, conf.v0)


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
        q, v, dv = sim.get_state()

        # # ########################
        # # tau_ff = pin.computeGeneralizedGravity(robot.model, robot.data, q)
        # tau_ff = pin.rnea(robot.model, robot.data, q, v, np.zeros(robot.nv))
        
        # # Pure feedforward
        # # tau = tau_ff
        # # PD
        # # tau = - Kp*(q - conf.q0) - Kd*(v - conf.v0)
        # # PD+
        # tau = tau_ff - Kp*(q - conf.q0) - Kd*(v - conf.v0)
        # # ########################

        #########################
        # Joint Space Inverse Dynamics
        ddqd = - Kp*(q - conf.q0) - Kd*(v - conf.v0)
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
            sim.apply_external_force(
                fext, frame_name, rf_frame=pin.LOCAL_WORLD_ALIGNED)

        sim.send_joint_command(tau)
        sim.step_simulation()

        delay = time.time() - t1
        if delay < dt_sim:
            # print(delay)
            time.sleep(dt_sim - delay)

    print('End')
