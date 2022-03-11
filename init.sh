#!/bin/sh

# run as root

gpsd /dev/ttyACM0
echo 0 > /sys/module/usbcore/parameters/usbfs_memory_mb
