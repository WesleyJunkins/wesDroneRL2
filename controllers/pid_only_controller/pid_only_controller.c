/*
 * Copyright 2022 Bitcraze AB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 *  ...........       ____  _ __
 *  |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 *  | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 *  | / ,..´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *     +.......   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 *
 * @file pid_controller.c
 * Description: A simple PID controller for attitude,
 *          height and velocity  control of an quadcopter
 * Author:      Kimberly McGuire (Bitcraze AB)

 */

 #include <math.h>
 #include <stdio.h>
 #include <string.h>
 
 #include "pid_controller.h"
 
 float constrain(float value, const float minVal, const float maxVal) {
   return fminf(maxVal, fmaxf(minVal, value));
 }
 
 double pastAltitudeError, pastPitchError, pastRollError, pastYawRateError;
 double pastVxError, pastVyError;
 double altitudeIntegrator;
 
 void init_pid_attitude_fixed_height_controller() {
   pastAltitudeError = 0;
   pastYawRateError = 0;
   pastPitchError = 0;
   pastRollError = 0;
   pastVxError = 0;
   pastVyError = 0;
   altitudeIntegrator = 0;
 }
 
 void pid_attitude_fixed_height_controller(actual_state_t actual_state, const desired_state_t *desired_state,
                                           gains_pid_t gains_pid, double dt, motor_power_t *motorCommands) {
   control_commands_t control_commands = {0};
   pid_fixed_height_controller(actual_state, desired_state, gains_pid, dt, &control_commands);
   pid_attitude_controller(actual_state, desired_state, gains_pid, dt, &control_commands);
   motor_mixing(control_commands, motorCommands);
 }
 
 void pid_velocity_fixed_height_controller(actual_state_t actual_state, desired_state_t *desired_state, gains_pid_t gains_pid,
                                           double dt, motor_power_t *motorCommands) {
   control_commands_t control_commands = {0};
   pid_horizontal_velocity_controller(actual_state, desired_state, gains_pid, dt);
   pid_fixed_height_controller(actual_state, desired_state, gains_pid, dt, &control_commands);
   pid_attitude_controller(actual_state, desired_state, gains_pid, dt, &control_commands);
   motor_mixing(control_commands, motorCommands);
 }
 
 void pid_fixed_height_controller(actual_state_t actual_state, const desired_state_t *desired_state, gains_pid_t gains_pid,
                                  double dt, control_commands_t *control_commands) {
   double altitudeError = desired_state->altitude - actual_state.altitude;
   double altitudeDerivativeError = (altitudeError - pastAltitudeError) / dt;
   control_commands->altitude =
     gains_pid.kp_z * constrain(altitudeError, -1, 1) + gains_pid.kd_z * altitudeDerivativeError + gains_pid.ki_z;
 
   altitudeIntegrator += altitudeError * dt;
   control_commands->altitude = gains_pid.kp_z * constrain(altitudeError, -1, 1) + gains_pid.kd_z * altitudeDerivativeError +
                                gains_pid.ki_z * altitudeIntegrator + 48;
   pastAltitudeError = altitudeError;
 }
 
 void motor_mixing(control_commands_t control_commands, motor_power_t *motorCommands) {
   // Motor mixing
   motorCommands->m1 = control_commands.altitude - control_commands.roll + control_commands.pitch + control_commands.yaw;
   motorCommands->m2 = control_commands.altitude - control_commands.roll - control_commands.pitch - control_commands.yaw;
   motorCommands->m3 = control_commands.altitude + control_commands.roll - control_commands.pitch + control_commands.yaw;
   motorCommands->m4 = control_commands.altitude + control_commands.roll + control_commands.pitch - control_commands.yaw;
 }
 
 void pid_attitude_controller(actual_state_t actual_state, const desired_state_t *desired_state, gains_pid_t gains_pid,
                              double dt, control_commands_t *control_commands) {
   // Calculate errors
   double pitchError = desired_state->pitch - actual_state.pitch;
   double pitchDerivativeError = (pitchError - pastPitchError) / dt;
   double rollError = desired_state->roll - actual_state.roll;
   double rollDerivativeError = (rollError - pastRollError) / dt;
   double yawRateError = desired_state->yaw_rate - actual_state.yaw_rate;
 
   // PID control
   control_commands->roll = gains_pid.kp_att_rp * constrain(rollError, -1, 1) + gains_pid.kd_att_rp * rollDerivativeError;
   control_commands->pitch = -gains_pid.kp_att_rp * constrain(pitchError, -1, 1) - gains_pid.kd_att_rp * pitchDerivativeError;
   control_commands->yaw = gains_pid.kp_att_y * constrain(yawRateError, -1, 1);
 
   // Save error for the next round
   pastPitchError = pitchError;
   pastRollError = rollError;
   pastYawRateError = yawRateError;
 }
 
 void pid_horizontal_velocity_controller(actual_state_t actual_state, desired_state_t *desired_state, gains_pid_t gains_pid,
                                         double dt) {
   double vxError = desired_state->vx - actual_state.vx;
   double vxDerivative = (vxError - pastVxError) / dt;
   double vyError = desired_state->vy - actual_state.vy;
   double vyDerivative = (vyError - pastVyError) / dt;
 
   // PID control
   double pitchCommand = gains_pid.kp_vel_xy * constrain(vxError, -1, 1) + gains_pid.kd_vel_xy * vxDerivative;
   double rollCommand = -gains_pid.kp_vel_xy * constrain(vyError, -1, 1) - gains_pid.kd_vel_xy * vyDerivative;
 
   desired_state->pitch = pitchCommand;
   desired_state->roll = rollCommand;
 
   // Save error for the next round
   pastVxError = vxError;
   pastVyError = vyError;
 }
 
/*
 * File:          pid_only_controller.c
 * Date:
 * Description:   Simple PID controller that alternates yaw to fly left and right
 * Author:
 * Modifications:
 */

#include <math.h>
#include <stdio.h>

#include <webots/gps.h>
#include <webots/gyro.h>
#include <webots/inertial_unit.h>
#include <webots/motor.h>
#include <webots/robot.h>

#define TIME_STEP 64
#define FLYING_ALTITUDE 1.0
#define YAW_ALTERNATION_PERIOD 3.0  // seconds to alternate yaw direction
#define YAW_RATE_MAGNITUDE 0.5      // magnitude of yaw rate command

/*
 * This is the main program.
 * The arguments of the main function can be specified by the
 * "controllerArgs" field of the Robot node
 */
int main(int argc, char **argv) {
  /* necessary to initialize webots stuff */
  wb_robot_init();

  const int timestep = (int)wb_robot_get_basic_time_step();

  // Initialize motors
  WbDeviceTag m1_motor = wb_robot_get_device("m1_motor");
  wb_motor_set_position(m1_motor, INFINITY);
  wb_motor_set_velocity(m1_motor, -1.0);
  WbDeviceTag m2_motor = wb_robot_get_device("m2_motor");
  wb_motor_set_position(m2_motor, INFINITY);
  wb_motor_set_velocity(m2_motor, 1.0);
  WbDeviceTag m3_motor = wb_robot_get_device("m3_motor");
  wb_motor_set_position(m3_motor, INFINITY);
  wb_motor_set_velocity(m3_motor, -1.0);
  WbDeviceTag m4_motor = wb_robot_get_device("m4_motor");
  wb_motor_set_position(m4_motor, INFINITY);
  wb_motor_set_velocity(m4_motor, 1.0);

  // Initialize sensors
  WbDeviceTag imu = wb_robot_get_device("inertial_unit");
  wb_inertial_unit_enable(imu, timestep);
  WbDeviceTag gps = wb_robot_get_device("gps");
  wb_gps_enable(gps, timestep);
  WbDeviceTag gyro = wb_robot_get_device("gyro");
  wb_gyro_enable(gyro, timestep);

  // Wait for 2 seconds to let the simulation stabilize
  while (wb_robot_step(timestep) != -1) {
    if (wb_robot_get_time() > 2.0)
      break;
  }

  // Initialize variables
  actual_state_t actual_state = {0};
  desired_state_t desired_state = {0};
  double past_x_global = 0;
  double past_y_global = 0;
  double past_time = wb_robot_get_time();

  // Initialize PID gains
  gains_pid_t gains_pid;
  gains_pid.kp_att_y = 1;
  gains_pid.kd_att_y = 0.5;
  gains_pid.kp_att_rp = 0.5;
  gains_pid.kd_att_rp = 0.1;
  gains_pid.kp_vel_xy = 2;
  gains_pid.kd_vel_xy = 0.5;
  gains_pid.kp_z = 10;
  gains_pid.ki_z = 5;
  gains_pid.kd_z = 5;
  init_pid_attitude_fixed_height_controller();

  // Initialize struct for motor power
  motor_power_t motor_power;

  // Yaw alternation variables
  double yaw_alternation_start_time = wb_robot_get_time();
  int yaw_direction = 1;  // 1 for positive yaw, -1 for negative yaw

  printf("Starting PID controller with alternating yaw...\n");

  /* main loop
   * Perform simulation steps and leave the loop when the simulation is over
   */
  while (wb_robot_step(timestep) != -1) {
    const double dt = wb_robot_get_time() - past_time;
    const double current_time = wb_robot_get_time();

    // Get measurements from sensors
    actual_state.roll = wb_inertial_unit_get_roll_pitch_yaw(imu)[0];
    actual_state.pitch = wb_inertial_unit_get_roll_pitch_yaw(imu)[1];
    actual_state.yaw_rate = wb_gyro_get_values(gyro)[2];
    actual_state.altitude = wb_gps_get_values(gps)[2];
    double x_global = wb_gps_get_values(gps)[0];
    double vx_global = (x_global - past_x_global) / dt;
    double y_global = wb_gps_get_values(gps)[1];
    double vy_global = (y_global - past_y_global) / dt;

    // Get body fixed velocities
    double actualYaw = wb_inertial_unit_get_roll_pitch_yaw(imu)[2];
    double cosyaw = cos(actualYaw);
    double sinyaw = sin(actualYaw);
    actual_state.vx = vx_global * cosyaw + vy_global * sinyaw;
    actual_state.vy = -vx_global * sinyaw + vy_global * cosyaw;

    // Check if it's time to alternate yaw direction
    if (current_time - yaw_alternation_start_time >= YAW_ALTERNATION_PERIOD) {
      yaw_direction *= -1;  // Flip direction
      yaw_alternation_start_time = current_time;
      printf("Alternating yaw direction: %s\n", yaw_direction > 0 ? "right" : "left");
    }

    // Set desired state
    desired_state.roll = 0;
    desired_state.pitch = 0;
    desired_state.vx = 0.2;  // Small forward velocity to move while yawing
    desired_state.vy = 0;
    desired_state.yaw_rate = yaw_direction * YAW_RATE_MAGNITUDE;  // Alternating yaw
    desired_state.altitude = FLYING_ALTITUDE;

    // Use PID velocity controller with fixed height
    pid_velocity_fixed_height_controller(actual_state, &desired_state, gains_pid, dt, &motor_power);

    // Set motor velocities
    wb_motor_set_velocity(m1_motor, -motor_power.m1);
    wb_motor_set_velocity(m2_motor, motor_power.m2);
    wb_motor_set_velocity(m3_motor, -motor_power.m3);
    wb_motor_set_velocity(m4_motor, motor_power.m4);

    // Save past values for next time step
    past_time = wb_robot_get_time();
    past_x_global = x_global;
    past_y_global = y_global;
  }

  /* This is necessary to cleanup webots resources */
  wb_robot_cleanup();

  return 0;
}
