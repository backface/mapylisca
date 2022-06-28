#!/bin/bash

HOST=$(cat /etc/hostname)
rsync -avc ./ /media/mash/rootfs/mapylisca-$HOST
