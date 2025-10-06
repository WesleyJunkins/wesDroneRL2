CONTROLLERS:
- pid_controller = old controller that is not used now
- wes_flight_controller_crazyflie = controller that takes discrete values and uses its own PID to stabilize the drone
- wes_flight_controller_crazyflie_nopid = controller that takes continuous values from any receiver and controls the drone

RECEIVERS:
- recvRuleDisc = rule-based discrete receiver that sends discrete commands (use with wes_flight_controller_crazyflie)
- testRecvRuleDisc = rule-based PID that creates continuous values and converts to discrete commands (use with wes_flight_controller_crazyflie)
- recvPID = rule-based PID that creates and sends continuous values (use with wes_flight_controller_crazyflie_nopid)