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
 
// No external PID controller needed - direct motor control
 
 #define FLYING_ALTITUDE 1.0
 #define SOCKET_PORT 8080
 #define BUFFER_SIZE 1024
 
// Structure to hold continuous control values from JSON
typedef struct {
    double forward_velocity;
    double lateral_velocity;
    double yaw_rate;
    double height_change;
    int reset_simulation;
} control_values_json_t;
 
 // Global socket for communication
 static int g_socket = -1;
 
// Function to parse JSON control values
control_values_json_t parse_control_json(const char* json_str) {
    control_values_json_t values = {0};
    
    // Simple JSON parsing for continuous values
    // Expected format: {"forward_velocity": 0.5, "lateral_velocity": -0.3, "yaw_rate": 0.8, "height_change": 0.1, "reset_simulation": 0}
    
    // Parse forward_velocity
    char* forward_start = strstr(json_str, "\"forward_velocity\":");
    if (forward_start) {
        forward_start = strchr(forward_start, ':');
        if (forward_start) {
            values.forward_velocity = atof(forward_start + 1);
        }
    }
    
    // Parse lateral_velocity
    char* lateral_start = strstr(json_str, "\"lateral_velocity\":");
    if (lateral_start) {
        lateral_start = strchr(lateral_start, ':');
        if (lateral_start) {
            values.lateral_velocity = atof(lateral_start + 1);
        }
    }
    
    // Parse yaw_rate
    char* yaw_start = strstr(json_str, "\"yaw_rate\":");
    if (yaw_start) {
        yaw_start = strchr(yaw_start, ':');
        if (yaw_start) {
            values.yaw_rate = atof(yaw_start + 1);
        }
    }
    
    // Parse height_change
    char* height_start = strstr(json_str, "\"height_change\":");
    if (height_start) {
        height_start = strchr(height_start, ':');
        if (height_start) {
            values.height_change = atof(height_start + 1);
        }
    }
    
    // Parse reset_simulation
    if (strstr(json_str, "\"reset_simulation\":1") || strstr(json_str, "\"reset_simulation\": 1")) {
        values.reset_simulation = 1;
    }
    
    return values;
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
 
// Function to check if values are non-zero (any control is active)
int is_non_idle_command(control_values_json_t values) {
    return (fabs(values.forward_velocity) > 0.01) || 
           (fabs(values.lateral_velocity) > 0.01) || 
           (fabs(values.yaw_rate) > 0.01) || 
           (fabs(values.height_change) > 0.01) || 
           values.reset_simulation;
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
 
// Function to check for incoming control values
control_values_json_t check_for_commands() {
    control_values_json_t values = {0};
    char buffer[BUFFER_SIZE];
    
    if (g_socket >= 0) {
        int bytes_received = recv(g_socket, buffer, BUFFER_SIZE - 1, 0);
        if (bytes_received > 0) {
            buffer[bytes_received] = '\0';
            values = parse_control_json(buffer);
        }
    }
    
    return values;
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
  double past_x_global = 0;
  double past_y_global = 0;
  double past_time = wb_robot_get_time();

  double height_desired = FLYING_ALTITUDE;
  
  // Track previous command state for image sending
  control_values_json_t prev_commands = {0};
 
   printf("\n");
 
   printf("====== Controls =======\n");
 
   printf(" The Crazyflie can be controlled from your keyboard!\n");
   printf(" All controllable movement is in body coordinates\n");
   printf("- Use the up, back, right and left button to move in the horizontal plane\n");
   printf("- Use Q and E to rotate around yaw\n ");
   printf("- Use W and S to go up and down\n");
   printf(" Socket communication enabled on port %d\n", SOCKET_PORT);
   
   //WES ADDED - Commands are now executed once per received command
 
  while (wb_robot_step(timestep) != -1) {
    const double dt = wb_robot_get_time() - past_time;

    // Get measurements
    double current_altitude = wb_gps_get_values(gps)[2];
    double x_global = wb_gps_get_values(gps)[0];
    double y_global = wb_gps_get_values(gps)[1];

    // Check for incoming control values
    control_values_json_t socket_values = check_for_commands();
     
     // Send ready-up 2D array 5 seconds after startup
     static int startup_counter = 0;
     startup_counter++;
     if (startup_counter == 500) {  // ~7 seconds after startup
         printf("Sending ready-up 2D array...\n");
         send_2d_array(camera);
     }
 
    // Check if we received any control values
    if (is_non_idle_command(socket_values)) {
        printf("Received control values: fwd=%.3f, lat=%.3f, yaw=%.3f, hgt=%.3f\n", 
               socket_values.forward_velocity, socket_values.lateral_velocity, 
               socket_values.yaw_rate, socket_values.height_change);
        
        if (socket_values.reset_simulation) {
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
        
        // Apply height change
        height_desired += socket_values.height_change * dt;
        
        // Direct motor control based on received values
        // Convert control values to motor speeds
        double base_speed = 100.0;  // Base motor speed
        double forward_factor = socket_values.forward_velocity;
        double lateral_factor = socket_values.lateral_velocity;
        double yaw_factor = socket_values.yaw_rate;
        double height_factor = (height_desired - current_altitude) * 10.0;  // Height control
        
        // Calculate individual motor speeds
        // Motor layout: m1=front-left, m2=front-right, m3=back-left, m4=back-right
        double m1_speed = base_speed + forward_factor * 50.0 + lateral_factor * 50.0 - yaw_factor * 30.0 + height_factor;
        double m2_speed = base_speed + forward_factor * 50.0 - lateral_factor * 50.0 + yaw_factor * 30.0 + height_factor;
        double m3_speed = base_speed + forward_factor * 50.0 - lateral_factor * 50.0 + yaw_factor * 30.0 + height_factor;
        double m4_speed = base_speed + forward_factor * 50.0 + lateral_factor * 50.0 - yaw_factor * 30.0 + height_factor;
        
        // Apply motor speeds
        wb_motor_set_velocity(m1_motor, -m1_speed);
        wb_motor_set_velocity(m2_motor, m2_speed);
        wb_motor_set_velocity(m3_motor, -m3_speed);
        wb_motor_set_velocity(m4_motor, m4_speed);
        
        // Wait a moment for the command to take effect, then send new array
        for (int i = 0; i < 10; i++) {
            wb_robot_step(timestep);
        }
        
        // Send new 2D array after executing command
        printf("Sending new 2D array after command execution...\n");
        send_2d_array(camera);
    }
 
    // Hover control when no commands received
    if (!is_non_idle_command(socket_values)) {
        // Simple hover control - maintain altitude
        double height_error = height_desired - current_altitude;
        double height_factor = height_error * 10.0;
        
        // Set hover motor speeds
        double hover_speed = 100.0 + height_factor;
        wb_motor_set_velocity(m1_motor, -hover_speed);
        wb_motor_set_velocity(m2_motor, hover_speed);
        wb_motor_set_velocity(m3_motor, -hover_speed);
        wb_motor_set_velocity(m4_motor, hover_speed);
    }
 
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
 