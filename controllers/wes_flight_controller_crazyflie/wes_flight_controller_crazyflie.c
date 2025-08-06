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

/*
 *  ...........       ____  _ __
 *  |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 *  | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 *  | / ,..´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *     +.......   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 *
 * @file crazyflie_controller.c
 * Description: Controls the crazyflie in webots
 * Author:      Kimberly McGuire (Bitcraze AB)
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <webots/camera.h>
#include <webots/distance_sensor.h>
#include <webots/gps.h>
#include <webots/gyro.h>
#include <webots/inertial_unit.h>
#include <webots/keyboard.h>
#include <webots/motor.h>
#include <webots/robot.h>

// Add external controller
#include "pid_controller.h"

#define FLYING_ALTITUDE 1.0
#define SOCKET_PORT 8080
#define BUFFER_SIZE 1024

// Structure to hold control commands from JSON
typedef struct {
    int forward;
    int backward;
    int left;
    int right;
    int yaw_increase;
    int yaw_decrease;
    int height_diff_increase;
    int height_diff_decrease;
    int reset_simulation;
} control_commands_json_t;

// Function to parse JSON control commands
control_commands_json_t parse_control_json(const char* json_str) {
    control_commands_json_t commands = {0};
    
    // More robust JSON parsing - handle spaces and different formats
    if (strstr(json_str, "\"forward\":1") || strstr(json_str, "\"forward\": 1")) commands.forward = 1;
    if (strstr(json_str, "\"backward\":1") || strstr(json_str, "\"backward\": 1")) commands.backward = 1;
    if (strstr(json_str, "\"left\":1") || strstr(json_str, "\"left\": 1")) commands.left = 1;
    if (strstr(json_str, "\"right\":1") || strstr(json_str, "\"right\": 1")) commands.right = 1;
    if (strstr(json_str, "\"yaw_increase\":1") || strstr(json_str, "\"yaw_increase\": 1")) commands.yaw_increase = 1;
    if (strstr(json_str, "\"yaw_decrease\":1") || strstr(json_str, "\"yaw_decrease\": 1")) commands.yaw_decrease = 1;
    if (strstr(json_str, "\"height_diff_increase\":1") || strstr(json_str, "\"height_diff_increase\": 1")) commands.height_diff_increase = 1;
    if (strstr(json_str, "\"height_diff_decrease\":1") || strstr(json_str, "\"height_diff_decrease\": 1")) commands.height_diff_decrease = 1;
    if (strstr(json_str, "\"reset_simulation\":1") || strstr(json_str, "\"reset_simulation\": 1")) commands.reset_simulation = 1;
    
    // Debug output to see what was parsed
    if (commands.forward || commands.backward || commands.left || commands.right || 
        commands.yaw_increase || commands.yaw_decrease || commands.height_diff_increase || 
        commands.height_diff_decrease || commands.reset_simulation) {
        printf("Parsed commands - forward:%d backward:%d left:%d right:%d yaw_inc:%d yaw_dec:%d height_inc:%d height_dec:%d reset:%d\n",
               commands.forward, commands.backward, commands.left, commands.right,
               commands.yaw_increase, commands.yaw_decrease, commands.height_diff_increase,
               commands.height_diff_decrease, commands.reset_simulation);
    }
    
    return commands;
}

// Global variable to store client socket
static int g_client_socket = -1;

// Function to handle socket communication
control_commands_json_t handle_socket_communication(int server_socket) {
    control_commands_json_t commands = {0};
    char buffer[BUFFER_SIZE];
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    
    // Set socket to non-blocking mode
    int flags = fcntl(server_socket, F_GETFL, 0);
    fcntl(server_socket, F_SETFL, flags | O_NONBLOCK);
    
    // If no client is connected, try to accept a new connection
    if (g_client_socket < 0) {
        g_client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_len);
        if (g_client_socket >= 0) {
            // Set client socket to non-blocking
            flags = fcntl(g_client_socket, F_GETFL, 0);
            fcntl(g_client_socket, F_SETFL, flags | O_NONBLOCK);
            printf("Client connected\n");
        }
    }
    
    // If we have a client connection, try to receive data
    if (g_client_socket >= 0) {
        int bytes_received = recv(g_client_socket, buffer, BUFFER_SIZE - 1, 0);
        if (bytes_received > 0) {
            buffer[bytes_received] = '\0';
            printf("Received JSON (%d bytes): %s\n", bytes_received, buffer);
            commands = parse_control_json(buffer);
        } else if (bytes_received == 0) {
            // Client disconnected
            printf("Client disconnected\n");
            close(g_client_socket);
            g_client_socket = -1;
        } else if (bytes_received < 0) {
            // No data available (non-blocking)
            // This is normal, just continue
        }
    } else {
        // No client connected - this is normal at startup
        static int connection_attempts = 0;
        connection_attempts++;
        if (connection_attempts % 1000 == 0) {  // Print every ~10 seconds
            printf("Waiting for client connection...\n");
        }
    }
    
    return commands;
}

// Function to initialize socket server
int init_socket_server() {
    int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket == -1) {
        printf("Failed to create socket\n");
        return -1;
    }
    
    // Set socket options to reuse address
    int opt = 1;
    setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(SOCKET_PORT);
    
    if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        printf("Failed to bind socket\n");
        close(server_socket);
        return -1;
    }
    
    if (listen(server_socket, 5) < 0) {
        printf("Failed to listen on socket\n");
        close(server_socket);
        return -1;
    }
    
    printf("Socket server initialized on port %d\n", SOCKET_PORT);
    return server_socket;
}

int main(int argc, char **argv) {
  wb_robot_init();

  const int timestep = (int)wb_robot_get_basic_time_step();

  // Initialize socket server
  int server_socket = init_socket_server();

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
  wb_keyboard_enable(timestep);
  WbDeviceTag gyro = wb_robot_get_device("gyro");
  wb_gyro_enable(gyro, timestep);
  WbDeviceTag camera = wb_robot_get_device("down_camera");
  wb_camera_enable(camera, timestep);
  WbDeviceTag range_front = wb_robot_get_device("range_front");
  wb_distance_sensor_enable(range_front, timestep);
  WbDeviceTag range_left = wb_robot_get_device("range_left");
  wb_distance_sensor_enable(range_left, timestep);
  WbDeviceTag range_back = wb_robot_get_device("range_back");
  wb_distance_sensor_enable(range_back, timestep);
  WbDeviceTag range_right = wb_robot_get_device("range_right");
  wb_distance_sensor_enable(range_right, timestep);

  // Wait for 2 seconds
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

  // Initialize PID gains.
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

  double height_desired = FLYING_ALTITUDE;

  // Initialize struct for motor power
  motor_power_t motor_power;

  printf("\n");

  printf("====== Controls =======\n");

  printf(" The Crazyflie can be controlled from your keyboard!\n");
  printf(" All controllable movement is in body coordinates\n");
  printf("- Use the up, back, right and left button to move in the horizontal plane\n");
  printf("- Use Q and E to rotate around yaw\n ");
  printf("- Use W and S to go up and down\n");
  printf(" Socket communication enabled on port %d\n", SOCKET_PORT);

  while (wb_robot_step(timestep) != -1) {
    const double dt = wb_robot_get_time() - past_time;

    // Get measurements
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

    // Initialize values
    desired_state.roll = 0;
    desired_state.pitch = 0;
    desired_state.vx = 0;
    desired_state.vy = 0;
    desired_state.yaw_rate = 0;
    desired_state.altitude = 1.0;

    double forward_desired = 0;
    double sideways_desired = 0;
    double yaw_desired = 0;
    double height_diff_desired = 0;

    // Handle socket communication and get control commands
    control_commands_json_t socket_commands = handle_socket_communication(server_socket);

    // Debug: Print socket command state periodically
    static int debug_counter = 0;
    debug_counter++;
    if (debug_counter % 1000 == 0) {  // Every ~10 seconds
        printf("Socket commands - forward:%d backward:%d left:%d right:%d yaw_inc:%d yaw_dec:%d height_inc:%d height_dec:%d reset:%d\n",
               socket_commands.forward, socket_commands.backward, socket_commands.left, socket_commands.right,
               socket_commands.yaw_increase, socket_commands.yaw_decrease, socket_commands.height_diff_increase,
               socket_commands.height_diff_decrease, socket_commands.reset_simulation);
    }

    // Control based on socket commands instead of keyboard
    if (socket_commands.forward) {
      printf("GO FORWARD - Setting forward_desired = +0.5\n");
      forward_desired = +0.5;
    }
    if (socket_commands.backward) {
      printf("GO BACKWARD - Setting forward_desired = -0.5\n");
      forward_desired = -0.5;
    }
    if (socket_commands.right) {
      printf("GO RIGHT - Setting sideways_desired = -0.5\n");
      sideways_desired = -0.5;
    }
    if (socket_commands.left) {
      printf("GO LEFT - Setting sideways_desired = +0.5\n");
      sideways_desired = +0.5;
    }
    if (socket_commands.yaw_increase) {
      printf("YAW INCREASE - Setting yaw_desired = 1.0\n");
      yaw_desired = 1.0;
    }
    if (socket_commands.yaw_decrease) {
      printf("YAW DECREASE - Setting yaw_desired = -1.0\n");
      yaw_desired = -1.0;
    }
    if (socket_commands.height_diff_increase) {
      printf("HEIGHT INCREASE - Setting height_diff_desired = 0.1\n");
      height_diff_desired = 0.1;
    }
    if (socket_commands.height_diff_decrease) {
      printf("HEIGHT DECREASE - Setting height_diff_desired = -0.1\n");
      height_diff_desired = -0.1;
    }
    if (socket_commands.reset_simulation) {
      // Reset simulation logic could be added here
      printf("Reset simulation requested\n");
    }

    height_desired += height_diff_desired * dt;

    // Example how to get sensor data
    // range_front_value = wb_distance_sensor_get_value(range_front));
    // const unsigned char *image = wb_camera_get_image(camera);

    desired_state.yaw_rate = yaw_desired;

    // PID velocity controller with fixed height
    desired_state.vy = sideways_desired;
    desired_state.vx = forward_desired;
    desired_state.altitude = height_desired;
    pid_velocity_fixed_height_controller(actual_state, &desired_state, gains_pid, dt, &motor_power);

    // Setting motorspeed
    wb_motor_set_velocity(m1_motor, -motor_power.m1);
    wb_motor_set_velocity(m2_motor, motor_power.m2);
    wb_motor_set_velocity(m3_motor, -motor_power.m3);
    wb_motor_set_velocity(m4_motor, motor_power.m4);

    // Reset command values to prevent accumulation from rapid socket commands
    socket_commands.forward = 0;
    socket_commands.backward = 0;
    socket_commands.left = 0;
    socket_commands.right = 0;
    socket_commands.yaw_increase = 0;
    socket_commands.yaw_decrease = 0;
    socket_commands.height_diff_increase = 0;
    socket_commands.height_diff_decrease = 0;
    socket_commands.reset_simulation = 0;

    // Save past time for next time step
    past_time = wb_robot_get_time();
    past_x_global = x_global;
    past_y_global = y_global;
  };

  // Clean up socket
  if (g_client_socket >= 0) {
    close(g_client_socket);
  }
  if (server_socket >= 0) {
    close(server_socket);
  }

  wb_robot_cleanup();

  return 0;
}
