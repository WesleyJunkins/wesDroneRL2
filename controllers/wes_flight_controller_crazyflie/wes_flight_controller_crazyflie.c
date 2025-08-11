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
#include <stdint.h>

#include <webots/camera.h>
#include <webots/distance_sensor.h>
#include <webots/gps.h>
#include <webots/gyro.h>
#include <webots/inertial_unit.h>
#include <webots/keyboard.h>
#include <webots/motor.h>
#include <webots/robot.h>
#include <webots/supervisor.h>

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
    int reset_values;
} control_commands_json_t;

// Global socket for communication
static int g_socket = -1;

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
    if (strstr(json_str, "\"reset_values\":1") || strstr(json_str, "\"reset_values\": 1")) commands.reset_values = 1;
    
    return commands;
}

// Function to convert RGBA image to grayscale brightness values (0-255)
int get_brightness(const unsigned char *rgba_pixel) {
    // Convert RGBA to grayscale using standard luminance formula
    // R*0.299 + G*0.587 + B*0.114
    int r = rgba_pixel[0];
    int g = rgba_pixel[1];
    int b = rgba_pixel[2];
    return (int)(r * 0.299 + g * 0.587 + b * 0.114);
}

// Function to convert brightness (0-255) to our 0-4 scale
int brightness_to_scale(int brightness) {
    if (brightness <= 51) return 0;      // Black
    if (brightness <= 102) return 1;     // Dark gray
    if (brightness <= 153) return 2;     // Medium gray
    if (brightness <= 204) return 3;     // Light gray
    return 4;                            // White
}

// Function to send 2D array to port
void send_2d_array(WbDeviceTag camera) {
    // Get camera image
    const unsigned char *image = wb_camera_get_image(camera);
    if (!image) {
        printf("Cannot send array: no image data available\n");
        return;
    }
    
    int width = wb_camera_get_width(camera);
    int height = wb_camera_get_height(camera);
    int channels = 4; // RGBA
    
    printf("Converting image to 2D array: %dx%d\n", width, height);
    
    // Create 2D array
    int8_t *array_2d = malloc(width * height * sizeof(int8_t));
    if (!array_2d) {
        printf("Failed to allocate memory for 2D array\n");
        return;
    }
    
    // Convert image to 2D array
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pixel_index = (y * width + x) * channels;
            int brightness = get_brightness(&image[pixel_index]);
            int scale_value = brightness_to_scale(brightness);
            array_2d[y * width + x] = (int8_t)scale_value;
        }
    }
    
    // Print the 2D array to console
    printf("\n2D Array (%dx%d) before sending:\n", width, height);
    printf("0=black, 1=dark_gray, 2=medium_gray, 3=light_gray, 4=white\n");
    printf("-");
    for (int i = 0; i < width * 2 + 8; i++) printf("-");
    printf("\n");
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%d ", array_2d[y * width + x]);
        }
        printf("\n");
    }
    
    printf("-");
    for (int i = 0; i < width * 2 + 8; i++) printf("-");
    printf("\n");
    
    // Send array data to port
    int array_size = width * height * sizeof(int8_t);
    send(g_socket, (const char*)array_2d, array_size, 0);
    printf("Sent 2D array: %dx%d, %d bytes\n", width, height, array_size);
    
    // Clean up
    free(array_2d);
}

// Function to check if commands are non-idle (any command is active)
int is_non_idle_command(control_commands_json_t commands) {
    return commands.forward || commands.backward || commands.left || commands.right ||
           commands.yaw_increase || commands.yaw_decrease || commands.height_diff_increase ||
           commands.height_diff_decrease || commands.reset_simulation || commands.reset_values;
}

// Function to initialize socket connection
int init_socket_connection() {
    g_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (g_socket == -1) {
        printf("Failed to create socket\n");
        return -1;
    }
    
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    server_addr.sin_port = htons(SOCKET_PORT);
    
    if (connect(g_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        printf("Failed to connect to port %d\n", SOCKET_PORT);
        close(g_socket);
        return -1;
    }
    
    // Set socket to non-blocking
    int flags = fcntl(g_socket, F_GETFL, 0);
    fcntl(g_socket, F_SETFL, flags | O_NONBLOCK);
    
    printf("Connected to port %d\n", SOCKET_PORT);
    return g_socket;
}

// Function to check for incoming commands
control_commands_json_t check_for_commands() {
    control_commands_json_t commands = {0};
    char buffer[BUFFER_SIZE];
    
    if (g_socket >= 0) {
        int bytes_received = recv(g_socket, buffer, BUFFER_SIZE - 1, 0);
        if (bytes_received > 0) {
            buffer[bytes_received] = '\0';
            commands = parse_control_json(buffer);
        }
    }
    
    return commands;
}

int main(int argc, char **argv) {
  wb_robot_init();

  const int timestep = (int)wb_robot_get_basic_time_step();

  // Initialize socket connection
  init_socket_connection();

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
  
  // Track previous command state for image sending
  control_commands_json_t prev_commands = {0};

  printf("\n");

  printf("====== Controls =======\n");

  printf(" The Crazyflie can be controlled from your keyboard!\n");
  printf(" All controllable movement is in body coordinates\n");
  printf("- Use the up, back, right and left button to move in the horizontal plane\n");
  printf("- Use Q and E to rotate around yaw\n ");
  printf("- Use W and S to go up and down\n");
  printf(" Socket communication enabled on port %d\n", SOCKET_PORT);
  
  //WES ADDED
  int constant_forward = 0;
  int constant_backward = 0;
  int constant_left = 0;
  int constant_right = 0;
  int constant_yaw_increase = 0;
  int constant_yaw_decrease = 0;
  int constant_height_diff_increase = 0;
  int constant_height_diff_decrease = 0;

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

    // Check for incoming commands
    control_commands_json_t socket_commands = check_for_commands();
    
    // Update previous commands
    prev_commands = socket_commands;

    // Send ready-up 2D array 5 seconds after startup
    static int startup_counter = 0;
    startup_counter++;
    if (startup_counter == 500) {  // ~7 seconds after startup
        printf("Sending ready-up 2D array...\n");
        send_2d_array(camera);
    }

    if (socket_commands.reset_simulation) {
      // Reset simulation using supervisor
      printf("Reset simulation requested - restarting simulation\n");
      
      // Get reference to the current robot node
      WbNodeRef robot_node = wb_supervisor_node_get_self();
      
      // Restart the simulation
      wb_supervisor_simulation_reset();
      
      // Restart this controller
      wb_supervisor_node_restart_controller(robot_node);
      
      return 0;  // Exit the current instance
    }
    
    //WES ADDED
    if(socket_commands.forward == 1) {
      constant_forward = 1;
    }
    if(socket_commands.backward == 1) {
      constant_backward = 1;
    }
    if(socket_commands.right == 1) {
      constant_right = 1;
    }
    if(socket_commands.left == 1) {
      constant_left = 1;
    }
    if(socket_commands.yaw_increase == 1) {
      constant_yaw_increase = 1;
    }
    if(socket_commands.yaw_decrease == 1) {
      constant_yaw_decrease = 1;
    }
    if(socket_commands.height_diff_increase == 1) {
      constant_height_diff_increase = 1;
    }
    if(socket_commands.height_diff_decrease == 1) {
      constant_height_diff_decrease = 1;
    }

    // Control based on socket commands instead of keyboard
    if (constant_forward == 1) {
      printf("GO FORWARD - Setting forward_desired = +0.5\n");
      forward_desired = +0.5;
    }
    if (constant_backward == 1) {
      printf("GO BACKWARD - Setting forward_desired = -0.5\n");
      forward_desired = -0.5;
    }
    if (constant_right == 1) {
      printf("GO RIGHT - Setting sideways_desired = -0.5\n");
      sideways_desired = -0.5;
    }
    if (constant_left == 1) {
      printf("GO LEFT - Setting sideways_desired = +0.5\n");
      sideways_desired = +0.5;
    }
    if (constant_yaw_increase == 1) {
      printf("YAW INCREASE - Setting yaw_desired = 1.0\n");
      yaw_desired = 1.0;
    }
    if (constant_yaw_decrease == 1) {
      printf("YAW DECREASE - Setting yaw_desired = -1.0\n");
      yaw_desired = -1.0;
    }
    if (constant_height_diff_increase == 1) {
      printf("HEIGHT INCREASE - Setting height_diff_desired = 0.1\n");
      height_diff_desired = 0.1;
    }
    if (constant_height_diff_decrease == 1) {
      printf("HEIGHT DECREASE - Setting height_diff_desired = -0.1\n");
      height_diff_desired = -0.1;
    }
    if (socket_commands.reset_values) {
      printf("Resetting values (Stopping flight temporarily)\n");
      constant_forward = 0;
      constant_backward = 0;
      constant_left = 0;
      constant_right = 0;
      constant_yaw_increase = 0;
      constant_yaw_decrease = 0;
      constant_height_diff_increase = 0;
      constant_height_diff_decrease = 0;
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

    // Save past time for next time step
    past_time = wb_robot_get_time();
    past_x_global = x_global;
    past_y_global = y_global;
  };

  // Clean up socket
  if (g_socket >= 0) {
    close(g_socket);
  }

  wb_robot_cleanup();

  return 0;
}
