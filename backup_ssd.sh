#!/bin/bash

HOST=$(cat /etc/hostname)
rsync -avc ./ /media/mash/b57640da-32cc-42fc-9ef8-cf640b479274/mapylisca-$HOST
