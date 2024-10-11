#!/bin/bash

# Change to the directory where the driver is located
cd /home/debian/OpenVLC/Latest_Version/Driver/

# Run the load.sh script with sudo privileges
sudo ./load_test.sh

# Navigate to the PRU TX directory
cd ..
cd /home/debian/OpenVLC/Latest_Version/PRU/RX

# Run the deploy.sh script with sudo privileges
sudo ./deploy.sh

# Return to the home directory
cd ~
