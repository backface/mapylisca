#!/usr/bin/env python3
#######################################
#
# my python line scanner (for Ximea)
#
# author:(c) Michael Aschauer <m AT ash.to>
# www: http:/m.ash.to
# licenced under: GPL v3
# see: http://www.gnu.org/licenses/gpl.html
#
#######################################

from ximea import xiapi
from threading import Thread
from ctypes import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
from gps import gps, WATCH_ENABLE
import mygeo
import time, datetime
import sys, os
import numpy as np
import screeninfo
import imageio
import copy
import cv2
import piexif
import math
import getopt
import json

from control.elphel353 import Camera as Elphel


# options
####################################

output_path = "scan-data/"
output_format = "avi"  # "jpg","jpeg","tif","tiff", "avi"
jpeg_quality = 95
line_height = 2
roi_height = 8
start_with_roi = False
initial_exposure = 0
initial_framerate = 500
output_height = 256
output_rotate = True
DEBUG_SPEED = True

config = {
  "output_path": "scan-data/",
  "output_format": "avi",
  "jpeg_quality": 95,
  "line_height": 2,
  "roi_height": 8,
  "initial_exposure": 0,
  "initial_framerate": 500,
  "output_height": 256,
  "start_with_roi": False,
  "start_with_autoconfig": False,
  "show_source": False,
  "output_rotate": True,
  "fullscreen": True,
  "ximea_device_id": None,
  "DEBUG_SPEED": False,
  "input": "ximea",
  "camcontrol": "ximea",
  "screenpos": "left"
}


#######################

config_file = None
output_size = (2064, config["output_height"])
input_size = (640, 480)
padding = 32
process = True

line_count = 1
frame_count = 0
line_index = int(input_size[1]/2) - int(config["line_height"]/2)
total_lines_count = 0
total_lines_index = 0
dist = 0

cam = None
img = None
data = None
scan_data = None
black_frame = None
last_full_frame = None
texture_id = 0
do_write_frame = False

time_start = 0
elapsed = 0
elapsed_total = 0
fps_timer_start = 0
last_print_time = 0
fps = 0
text = ''
text_gps = ''

thread_quit = 0
videowriter = None
video_thread = None
gpsd_thread = None
gpsd = None
tile_buffer = []
tile_size = None
tile_texture_ids = []
do_shift_tiles = True
zoom_in = False

screen_id = 0
screen = screeninfo.get_monitors()[screen_id]
preview_size = [screen.height - 32, screen.height - 32]

drag_exp = False
drag_fps = False
last_drag_time = 0
slider_exp_pos = 0
slider_fps_pos = 0
button_ae_pos_y = preview_size[1] * 1 / 5
button_wb_pos_y = preview_size[1] * 2 / 5
button_scan_pos_y = preview_size[1] * 3 / 5
button_input_pos_y = preview_size[1] * 4 / 5
buttons_pos_x = 50

cam_is_ae = False
cam_is_wb = False
cam_fps = 0
cam_exp = 0
cam_gain = 0

mouse_pos = [0,0]



def usage():
	print("""
usage: mapylisca.py [options]

linescanner in python

options:
    -h, --help                  print usage
    -c, --configfile=FILE       config file
""")

def process_args():
	global config_file
	global verbose

	try:
		opts, args = getopt.getopt(sys.argv[1:], "hc:",
			["help", "configfile=",])
	except getopt.GetoptError:
		# print help information and exit:
		print("wrong options") # will print something like "option -a not recognized"
		usage()
		sys.exit(2)

	for o, a in opts:
		if o in ("-h", "--help"):
			usage()
			sys.exit()
		elif o in ("-c", "--config_file"):
			config_file = a
		else:
			assert False, "unhandled option"


def init():
  global config
  global cam, img, process
  global elphel
  global data, scan_data
  global input_size, output_size
  global full_size
  global fps, fps_timer_start, time_start
  global videowriter
  global video_thread
  global tile_buffer
  global black_frame
  global last_full_frame
  global scanlog_file, scanlog_file_single
  global config_file
  global output_isVideo

  if config_file:
    if not os.path.exists(config_file):
      print("can't find config")
      sys.exit()
  else:
    if os.path.exists("mapylisca.json"):
      config_file = "mapylisca.json"
    elif os.path.exists("~/mapylisca.json"):
     config_file = "~/mapylisca.json"

  if config_file:
    print("using config file:", config_file)
    f = open(config_file)
    data = json.load(f)
    for key in data:
      config[key] = data[key]
    f.close()
  else:
    print("don't use config file")

  output_isVideo = config["output_format"].lower() in ["mjpg", "mjpeg", "avi"]

  print("current config is:")
  print(json.dumps(config, indent=4))
  print("-------------------")

  # open camera
  if config["input"] == "ximea":
    cam = xiapi.Camera()
    if config["ximea_device_id"]:
      print('Opening camera id:', config["ximea_device_id"])
      cam.open_device_by_SN(config["ximea_device_id"])
    else:
      print('Opening first camera...')
      cam.open_device()

    # setup camera
    cam.set_imgdataformat('XI_RGB32')

    if config["initial_exposure"]:
      cam.set_exposure(config["initial_exposure"])
    else:
      cam.enable_auto_wb()
      cam.enable_aeag()
    cam.set_framerate(config["initial_framerate"])
    cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE_LIMIT")
    img = xiapi.Image()

    print('Starting data acquisition...')
    cam.start_acquisition()

    # get first image and set dimensions
    cam.get_image(img)
    input_size = (img.width, img.height)
    full_size = input_size

    if config["start_with_autoconfig"]:
      counter = 0
      read_frames = 25
      while (counter < read_frames):
        cam.get_image(img)
        counter = counter + 1
        print("reading frame #{}/{}".format(counter, read_frames))
      print("now go into scanning mode")
      config["start_with_roi"] = True


    # set up roi scan mode
    if config["start_with_roi"]:
      cam.stop_acquisition()
      cam.set_height(config["roi_height"])
      cam.set_offsetY(int(full_size[1]/2))
      cam.start_acquisition()
      cam.get_image(img)
      input_size = (img.width, img.height)
      cam.disable_auto_wb()
      cam.disable_aeag()

    # print out camera infos
    print("cam infos")
    print("=========================")
    print("image size", img.width, img.height)
    print("shutter type:", cam.get_shutter_type())
    print("white balance is auto:", cam.is_auto_wb()) # enable_auto_wb
    print("automatic exposure/gain:", cam.is_aeag()) # enable_aeg
    print("get_gain:", cam.get_gain())
    print("get_exposure:", cam.get_exposure())
    print("get_offsetY:", cam.get_offsetY())
    print("set_height :", cam.get_height())
    print("is_iscooled :", cam.is_iscooled())
    #print("get_temp :", cam.get_temp())
    #print("get_chip_temp :", cam.get_chip_temp())
    print("get_framerate :", cam.get_framerate())
    print("get_acq_timing_mode", cam.get_acq_timing_mode())
    print("get_limit_bandwidth_mode",  cam.get_limit_bandwidth_mode())
    print("get_offsetY_increment", cam.get_offsetY_increment())
    print("=========================")

    output_size = (img.width, output_size[1])

  elif config["input"] == "gstreamer":
    cam = cv2.VideoCapture(config["pipeline"], cv2.CAP_GSTREAMER)
    ret,frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    input_size = (frame.shape[1], frame.shape[0])
    full_size = input_size
    output_size = (frame.shape[1], output_size[0])
    print(output_size)

  if config["camcontrol"] == "elphel":
    elphel = Elphel(config["ip"])
    print(output_size)

  if config["camcontrol"] == "none":
    cam = cv2.VideoCapture(config["pipeline"], cv2.CAP_GSTREAMER)
    ret,frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    input_size = (frame.shape[1], frame.shape[0])
    full_size = input_size
    output_size = (frame.shape[1], config["output_height"])
    print(output_size)
    
  black_frame = np.zeros((output_size[1], output_size[0], 4), dtype=c_ubyte)
  scan_data = copy.deepcopy(black_frame)
  last_full_frame = copy.deepcopy(black_frame)

  # start timers
  time_start = time.time()
  fps_timer_start =  time.time()

  # start storing scan results as video
  outfile = '{}/{}.avi'.format(config["output_path"], time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
  os.makedirs(os.path.dirname(outfile), exist_ok=True)
  scanlog_filename = outfile[:-4] + ".log"
  scanlog_file= open(scanlog_filename, "w", buffering=1)

  if not output_isVideo:
    config["output_path"] = '{}/{}'.format(config["output_path"], time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    os.makedirs(config["output_path"], exist_ok=True)
    scanlog_filename = '{}/scan-{:06.0f}.{}'.format(config["output_path"], frame_count+1, "log")
    scanlog_file_single = open(scanlog_filename, "w", buffering=1)
  else:
    scanlog_file_single = None
    videowriter = imageio.get_writer(outfile,
      format='FFMPEG',
      mode='I',
      fps=25,
      codec='mjpeg',
      output_params=['-q:v','1'],
      #pixelformat='yuvj420p'
    )


def init_gl(width, height):
  global texture_id

  glClearColor(0,0,0, 1.0)
  glClearDepth(1.0)
  #glDepthFunc(GL_LESS)
  #glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  #glEnable(GL_DEPTH_TEST)
  glDisable(GL_CULL_FACE);
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  glOrtho(0, float(width), 0, float(height), -1, 1);
  # gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)
  glMatrixMode(GL_MODELVIEW)
  glEnable(GL_TEXTURE_2D)
  # create texture
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  texture_id = glGenTextures(1)


def update_frame():
  global config
  global cam, img, text
  global data, scan_data
  global input_size, full_size, output_size
  global fps, fps_timer_start, elapsed, elapse_total, line_count
  global do_shift_tiles, do_write_frame
  global blackframe
  global last_full_frame
  global frame_count
  global videowriter
  global elapsed_total
  global total_lines_count, total_lines_index
  global do_write_frame
  global gpsd
  global scanlog_file, scanlog_file_single
  global output_isVideo

  line_input_index = int(input_size[1]/2) - int(config["line_height"]/2)
  line_output_index = 0
  elapsed_total = 0

  while(True):
    if process:
      start_frame = time.time()

      if config["input"] == "ximea":
        cam.get_image(img)
        data = img.get_image_data_numpy()
      elif config["input"] == "gstreamer":
        ret,frame = cam.read()
        if ret:
          data = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        else:
          print("video ended.")
          do_write_frame = True
          quit()

      line_input_index = int(input_size[1]/2) - int(config["line_height"]/2)


      for i in range(0, config["line_height"]):
        scan_data[line_output_index,:] = data[line_input_index + i,:]
        # shift array (expensive)
        # scan_data = np.append(scan_data, [data[line_index + i,:]], 0)
        # scan_data = np.delete(scan_data, 0, 0)
        line_output_index = line_output_index + 1
        if line_output_index >= output_size[1]:
          last_full_frame = copy.deepcopy(scan_data)
          do_shift_tiles = True
          do_write_frame = True
          scan_data = black_frame.copy()
          line_output_index = 0
          frame_count = frame_count + 1
          if config["DEBUG_SPEED"]:
            print('-----------------------------')
            print('finished frame ...')
            print('-----------------------------')
      line_count = line_count + 1
      total_lines_count = total_lines_count + 1
      total_lines_index = line_output_index

      if (line_count % 25 == 0):
        end = time.time()
        seconds = end - fps_timer_start
        fps  = 25 / seconds
        fps_timer_start = end

      if config["DEBUG_SPEED"]:
        print(' #{:0.0f}/#{:0.0f} - frame render time: {:2.2f}ms '.format(line_count, frame_count, elapsed * 1000))
      elapsed = time.time() - start_frame
      elapsed_total = time.time() - time_start

    if do_write_frame:
      frame_out = copy.deepcopy(last_full_frame)
      # write as single images
      if not output_isVideo:
        # write exif tags
        zeroth_ifd = {
          piexif.ImageIFD.Artist: u"Michael Aschauer",
          piexif.ImageIFD.Make: "XIMEA",  # ASCII, count any
          piexif.ImageIFD.XResolution: (72, 1),
          piexif.ImageIFD.YResolution: (72, 1),
          piexif.ImageIFD.Software: u"mapylisca"
        }
        exif_ifd = {
          piexif.ExifIFD.DateTimeOriginal: time.strftime("%Y:%m:%d %H:%M:%S", time.gmtime())
        }
        #gps exif data
        if gpsd:
          lat_deg = mygeo.toDegMinSec(gpsd.fix.latitude)
          lng_deg = mygeo.toDegMinSec(gpsd.fix.longitude)
          gps_ifd = {
            piexif.GPSIFD.GPSLatitude: (mygeo.toRational(lat_deg[0]), mygeo.toRational(lat_deg[1]), mygeo.toRational(lat_deg[2])),
            piexif.GPSIFD.GPSLongitude: (mygeo.toRational(lng_deg[0]), mygeo.toRational(lng_deg[1]), mygeo.toRational(lng_deg[2])),
            piexif.GPSIFD.GPSLatitudeRef: mygeo.toLatRef(gpsd.fix.latitude),
            piexif.GPSIFD.GPSLongitudeRef: mygeo.toLonRef(gpsd.fix.longitude),
            piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
          }
          exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": gps_ifd }
        else:
          exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd }
        exif_bytes = piexif.dump(exif_dict)

        outfile = '{}/scan-{:06.0f}.{}'.format(config["output_path"], frame_count, config["output_format"])

        if config["output_format"].lower() in ["jpg", "jpeg"]:
          if config["output_rotate"]:
            imageio.imwrite(outfile, cv2.cvtColor(cv2.rotate(frame_out, cv2.ROTATE_90_COUNTERCLOCKWISE),  cv2.COLOR_RGB2BGR), quality=config["jpeg_quality"], exif = exif_bytes)
          else:
            imageio.imwrite(outfile, cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB), quality=config["jpeg_quality"], exif = exif_bytes)
        else:
          if config["output_rotate"]:
            imageio.imwrite(outfile, cv2.cvtColor(cv2.rotate(frame_out, cv2.ROTATE_90_COUNTERCLOCKWISE),  cv2.COLOR_RGB2BGR))
          else:
            imageio.imwrite(outfile, cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB))
        print("saving: {}".format(outfile))

        scanlog_file_single.close()
        scanlog_filename = '{}/scan-{:06.0f}.{}'.format(config["output_path"], frame_count+1, "log")
        scanlog_file_single= open(scanlog_filename, "w", buffering=1)
        write_logline(False)

      #write as video frames
      else:
        if config["output_rotate"]:
          videowriter.append_data(cv2.cvtColor(cv2.rotate(frame_out, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.COLOR_RGB2BGR))
        else:
          videowriter.append_data(cv2.cvtColor(frame_out), cv2.COLOR_RGB2BGR)
      do_write_frame = False

    if thread_quit:
      break
  print("")
  if config["input"] == "ximea":
    cam.stop_acquisition
    print("stopped camera acquisition")
    cam.close_device()
    print("closed device")
  elif config["input"] == "gstreamer":
    cam.release()


def update_write_frame():
  global last_full_frame
  global thread_quit
  global videowriter
  global do_write_frame
  global gpsd
  global scanlog_file, scanlog_file_single
  global output_isVideo

  while(True):
    if do_write_frame:
      frame_out = copy.deepcopy(last_full_frame)
      # write as single images
      if not output_isVideo:
        # write exif tags
        zeroth_ifd = {
          piexif.ImageIFD.Artist: u"Michael Aschauer",
          piexif.ImageIFD.Make: "XIMEA",  # ASCII, count any
          piexif.ImageIFD.XResolution: (72, 1),
          piexif.ImageIFD.YResolution: (72, 1),
          piexif.ImageIFD.Software: u"mapylisca"
        }
        exif_ifd = {
          piexif.ExifIFD.DateTimeOriginal: time.strftime("%Y:%m:%d %H:%M:%S", time.gmtime())
        }
        #gps exif data
        if gpsd:
          lat_deg = mygeo.toDegMinSec(gpsd.fix.latitude)
          lng_deg = mygeo.toDegMinSec(gpsd.fix.longitude)
          gps_ifd = {
            piexif.GPSIFD.GPSLatitude: (mygeo.toRational(lat_deg[0]), mygeo.toRational(lat_deg[1]), mygeo.toRational(lat_deg[2])),
            piexif.GPSIFD.GPSLongitude: (mygeo.toRational(lng_deg[0]), mygeo.toRational(lng_deg[1]), mygeo.toRational(lng_deg[2])),
            piexif.GPSIFD.GPSLatitudeRef: mygeo.toLatRef(gpsd.fix.latitude),
            piexif.GPSIFD.GPSLongitudeRef: mygeo.toLonRef(gpsd.fix.longitude),
            piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
          }
          exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": gps_ifd }
        else:
          exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd }
        exif_bytes = piexif.dump(exif_dict)

        outfile = '{}/scan-{:06.0f}.{}'.format(config["output_path"], frame_count, config["output_format"])

        if config["output_format"].lower() in ["jpg", "jpeg"]:
          if config["output_rotate"]:
            imageio.imwrite(outfile, cv2.cvtColor(cv2.rotate(frame_out, cv2.ROTATE_90_COUNTERCLOCKWISE),  cv2.COLOR_RGB2BGR), quality=config["jpeg_quality"], exif = exif_bytes)
          else:
            imageio.imwrite(outfile, cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB), quality=config["jpeg_quality"], exif = exif_bytes)
        else:
          if config["output_rotate"]:
            imageio.imwrite(outfile, cv2.cvtColor(cv2.rotate(frame_out, cv2.ROTATE_90_COUNTERCLOCKWISE),  cv2.COLOR_RGB2BGR))
          else:
            imageio.imwrite(outfile, cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB))
        print("saving: {}".format(outfile))

        scanlog_file_single.close()
        scanlog_filename = '{}/scan-{:06.0f}.{}'.format(config["output_path"], frame_count+1, "log")
        scanlog_file_single= open(scanlog_filename, "w", buffering=1)
        write_logline(False)

      #write as video frames
      else:
        if config["output_rotate"]:
          videowriter.append_data(cv2.cvtColor(cv2.rotate(frame_out, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.COLOR_RGB2BGR))
        else:
          videowriter.append_data(cv2.cvtColor(frame_out), cv2.COLOR_RGB2BGR)
      do_write_frame = False

    if thread_quit:
      break
    time.sleep (0.01)
  if output_isVideo:
    videowriter.close()
  print("stopped writing file")


def draw_gl_scene():
  global texture_id
  global thread_quit
  global line_count
  global tile_buffer
  global tile_size
  global tile_texture_ids
  global do_shift_tiles
  global last_full_frame
  global black_frame
  global scan_data
  global frame_count
  global last_print_time
  global elapsed_total
  global text
  global text_gps
  global gpsd, dist
  global mouse_pos
  global zoom_in
  global slider_exp_pos, slider_fps_pos
  global button_ae_pos_y, button_input_pos_y, button_scan_pos_y, button_wb_pos_y
  global cam_is_ae, cam_is_wb, cam_fps, cam_exp, cam_gain

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
  glLoadIdentity()

  ratio = 1
  if (config["show_source"]):
    frame = data
    ratio = input_size[1]/input_size[0]
  else:
    frame = scan_data
    if not tile_size:
      ratio = (preview_size[1] - 3 * padding)/output_size[0]
      tile_size = (round(output_size[0] * ratio), round(output_size[1] * ratio))

  # prepare scan frame texture
  # tx_image = cv2.flip(frame, 0)
  tx_image = Image.fromarray(frame)
  ix = tx_image.size[0]
  iy = tx_image.size[1]
  tx_image = tx_image.tobytes('raw', 'BGRA', 0, -1)
  glEnable(GL_TEXTURE_2D);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, tx_image)
  glBindTexture(GL_TEXTURE_2D, texture_id)

  # setup tile textures
  if len(tile_texture_ids) == 0:
    print('setup tiles')
    num_tiles = max(1, int(preview_size[1] / (output_size[1] * preview_size[0]/output_size[0])))
    if num_tiles > 1:
      tile_texture_ids = glGenTextures(num_tiles)
    else:
      tile_texture_ids = [ glGenTextures(num_tiles)]
    black = Image.fromarray(black_frame)
    tile = black.tobytes('raw', 'BGRA', 0, -1)
    for i in range(0, num_tiles):
      tile_buffer.append(tile)
      ix = black.size[0]
      iy = black.size[1]
      glBindTexture(GL_TEXTURE_2D, tile_texture_ids[i])
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE,  tile_buffer[i] )

  # show input souce
  if (config["show_source"]):
    ratio = input_size[1]/input_size[0]
    glPushMatrix()
    glTranslatef(0.0, 0.0, 0.0);
    glRotatef(90, 0, 0, 1);
    glBegin(GL_QUADS);
    if zoom_in:
      glTexCoord2f(0, 0); glVertex3f(-(full_size[1] - 2* padding), - (full_size[1] - 2* padding) * ratio, 0);
      glTexCoord2f(1, 0); glVertex3f( full_size[1] - 2* padding, - (full_size[1] - 2* padding) * ratio, 0);
      glTexCoord2f(1, 1); glVertex3f( full_size[1] - 2* padding, (full_size[1] - 2* padding) * ratio, 0);
      glTexCoord2f(0, 1); glVertex3f(-(full_size[1] - 2* padding),  (full_size[1] - 2* padding) * ratio, 0);
    else:
      glTexCoord2f(0, 0); glVertex3f(-(preview_size[1] - 2* padding), - (preview_size[1] - 2* padding) * ratio, 0);
      glTexCoord2f(1, 0); glVertex3f( preview_size[1] - 2* padding, - (preview_size[1] - 2* padding) * ratio, 0);
      glTexCoord2f(1, 1); glVertex3f( preview_size[1] - 2* padding, (preview_size[1] - 2* padding) * ratio, 0);
      glTexCoord2f(0, 1); glVertex3f(-(preview_size[1] - 2* padding),  (preview_size[1] - 2* padding) * ratio, 0);
    glEnd();
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glPopMatrix()

  # show scan result
  else:
    # shift tiles if new frame has arrived
    if do_shift_tiles:
      for i, buffer_id in enumerate(tile_texture_ids):
        if i < len(tile_texture_ids) - 1:
          tile_buffer[i] = copy.deepcopy(tile_buffer[i+1])
        else:
          tile_buffer[i] = Image.fromarray(last_full_frame).tobytes('raw', 'BGRA', 0, -1)
        glBindTexture(GL_TEXTURE_2D, buffer_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE,  tile_buffer[i] )
      do_shift_tiles = False

    # show tile buffers
    for i, buffer_id in enumerate(tile_texture_ids):
      glPushMatrix()
      glTranslatef((preview_size[1] - padding) - ((len(tile_texture_ids) - i + 1) * 2 * tile_size[1] - tile_size[1]), 0.0, 0.0);
      glRotatef(90, 0, 0, 1);
      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, buffer_id)
      glBegin(GL_QUADS);
      glTexCoord2f(0, 0); glVertex3f(-tile_size[0], -tile_size[1], 0);
      glTexCoord2f(1, 0); glVertex3f( tile_size[0], -tile_size[1], 0);
      glTexCoord2f(1, 1); glVertex3f( tile_size[0],  tile_size[1], 0);
      glTexCoord2f(0, 1); glVertex3f(-tile_size[0],  tile_size[1], 0);
      glEnd();
      glPopMatrix()
    # show main scan frame
    glPushMatrix()
    glTranslatef((preview_size[1] - padding) - tile_size[1], 0.0, 0.0);
    glRotatef(90, 0, 0, 1);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex3f(-tile_size[0], -tile_size[1], 0);
    glTexCoord2f(1, 0); glVertex3f( tile_size[0], -tile_size[1], 0);
    glTexCoord2f(1, 1); glVertex3f( tile_size[0],  tile_size[1], 0);
    glTexCoord2f(0, 1); glVertex3f(-tile_size[0],  tile_size[1], 0);
    glEnd();
    glPopMatrix()

  # draw lines
  glPushMatrix()
  glDisable(GL_TEXTURE_2D);
  glTranslatef(0.0, 0.0, -0.1)
  glColor4f(1.0, 1.0, 1.0, 1.0)
  glBegin(GL_LINES)
  glVertex3f(-preview_size[0], 0.0, 0.0);
  glVertex3f( preview_size[0], 0.0, 0.0);
  glEnd();
  glPopMatrix()

  # draw lines
  glPushMatrix()
  glDisable(GL_TEXTURE_2D);
  glTranslatef(0.0, 0.0, 0.1)
  glColor4f(1.0, 1.0, 1.0, 1.0)
  glBegin(GL_LINES)
  glVertex3f(-preview_size[0],  preview_size[1] - 60, 0.0);
  glVertex3f( preview_size[0],  preview_size[1] - 60, 0.0);
  glVertex3f(-preview_size[0], -preview_size[1] + 60, 0.0);
  glVertex3f( preview_size[0], -preview_size[1] + 60, 0.0);
  glEnd();
  glPopMatrix()

  # draw text info
  if (time.time() - last_print_time > 0.5):
    last_print_time = time.time()
    if not thread_quit:
      cam_exp = getExposure()
      cam_fps = getFramerate()
      cam_is_ae = getAE()
      cam_is_wb = getAWB()
      cam_gain = getGain()

      slider_exp_pos = microseconds2x(cam_exp)
      slider_fps_pos = fps2x(cam_fps)
      text = '{:02.0f}:{:02.0f}:{:02.0f} | LH={:0.0f} | #{:0.0f}/#{:0.0f} | FPS: {:3.0f}/{:3.0f} (real/req) | EXP: {:2.2f}ms ({:1.0f}db) | AE={:0.0f} WB={:0.0f} | {:2.2f}ms '.format(
        (elapsed_total/3600.0),  (elapsed_total/60) % 60, (elapsed_total % 60),
        config["line_height"],
        frame_count,
        line_count,
        fps,
        cam_fps,
        cam_exp/1000,
        cam_gain,
        cam_is_ae,
        cam_is_wb,
        elapsed * 1000
      )
    if not gpsd:
      text_gps = 'GPS: NA'
    elif gpsd.fix.mode < 2:
      text_gps = "GPS: NO FIX"
    else:
      text_gps = 'GPS: M={}, S={}, LAT={:02.6f}, LON={:02.6f}, ALT={:03.1f}, SPEED={:02.1f} km/h, DIST={:.2f}km'.format(
        gpsd.fix.mode,
        len(list(filter(lambda x: x.used, gpsd.satellites))),
        gpsd.fix.latitude,
        gpsd.fix.longitude,
        gpsd.fix.altitude,
        gpsd.fix.speed * 3.6, # m/s to km/h
        dist / 1000
      )
      #print('\r' + text, end=" ... ")

  glPushMatrix()
  glTranslatef(0.0, 0.0, -0.1)
  gl_write(text, - len(text) * 9, preview_size[1] - padding - 12)
  gl_write(text_gps, - len(text) * 9, -(preview_size[1] - padding + 3))
  glDisable(GL_TEXTURE_2D);
  glPopMatrix()



  '''
  glPushMatrix()
  glDisable(GL_TEXTURE_2D);
  glRasterPos(mouse_pos['x'], mouse_pos['y'])
  glBegin(GL_QUADS)
  glColor3f(1, 1, 0)
  glVertex2i(-0.1,  0.1)
  glVertex2i( 0.1,  0.1)
  glVertex2i( 0.1, -0.1)
  glVertex2i(-0.1, -0.1)
  glVertex2i(-0.1,  0.1)
  glEnd();
  glPopMatrix()
  '''

  # draw a triangle for expose
  glPushMatrix()
  glDisable(GL_TEXTURE_2D);
  glTranslatef(-(preview_size[0]) + 2 * slider_exp_pos - 20, preview_size[1] - 60, 0.1)
  glColor4f(1.0, 1.0, 1.0, 1.0)
  glBegin(GL_TRIANGLES)
  glVertex3f(-20, 0, 0.0)
  glVertex3f( 20, 0, 0.0)
  glVertex3f(  0, -30, 0.0)
  glEnd();
  glPopMatrix()

  # draw a triangle for framerate
  glPushMatrix()
  glDisable(GL_TEXTURE_2D);
  glTranslatef(-(preview_size[0]) + 2 * slider_fps_pos - 20, -preview_size[1] + 60, 0.1)
  glColor4f(1.0, 1.0, 1.0, 1.0)
  glBegin(GL_TRIANGLES)
  glVertex3f(-20,  0, 0.0)
  glVertex3f( 20,  0, 0.0)
  glVertex3f(  0, 30, 0.0)
  glEnd();
  glPopMatrix()

  # draw ae button
  glPushMatrix()
  glDisable(GL_TEXTURE_2D);
  glDepthMask(False);
  glTranslatef(-(preview_size[0]) + 2 * buttons_pos_x, (preview_size[1]) - 2 * button_ae_pos_y, 0.1)
  glColor4f(1.0, 1.0, 1.0, 0.5)
  if cam_is_ae:
    glBegin(GL_QUADS)
  else:
    glBegin(GL_LINE_LOOP)
  glVertex3f(-50, 50, 0.0)
  glVertex3f( 50, 50, 0.0)
  glVertex3f( 50,-50, 0.0)
  glVertex3f(-50,-50, 0.0)
  glEnd();
  if cam_is_ae:
    glColor4f(1.0, 0.0, 0, 1.0)
  glTranslatef(0, 0, 0.1)
  gl_write_big('AE')
  glDepthMask(True);

  glPopMatrix()


  # draw wb button
  glPushMatrix()
  glDisable(GL_TEXTURE_2D);
  glTranslatef(-(preview_size[0]) + 2 * buttons_pos_x,  (preview_size[1]) - 2 * button_wb_pos_y, 0.1)
  glColor4f(1.0, 1.0, 1.0, 0.5)
  if cam_is_wb:
    glBegin(GL_QUADS)
  else:
    glBegin(GL_LINE_LOOP)
  glVertex3f(-50, 50, 0.0)
  glVertex3f( 50, 50, 0.0)
  glVertex3f( 50,-50, 0.0)
  glVertex3f(-50,-50, 0.0)
  glEnd();
  if cam_is_wb:
    glColor4f(1.0, 0.0, 0, 1.0)
  glTranslatef(0, 0, 0.1)
  gl_write_big('WB')
  glPopMatrix()

  # draw source button
  glPushMatrix()
  glDisable(GL_TEXTURE_2D);
  glTranslatef(-(preview_size[0]) + 2 * buttons_pos_x, (preview_size[1]) - 2 * button_input_pos_y, 0.1)
  glColor4f(1.0, 1.0, 1.0, 0.5)
  if not config["show_source"]:
    glBegin(GL_QUADS)
  else:
    glBegin(GL_LINE_LOOP)
  glVertex3f(-50, 50, 0.0)
  glVertex3f( 50, 50, 0.0)
  glVertex3f( 50,-50, 0.0)
  glVertex3f(-50,-50, 0.0)
  glEnd()
  glTranslatef(0, 0, 0.1)
  if not config["show_source"]:
    glColor4f(1.0, 0.0, 0, 1.0)
    gl_write_big('SCAN')
  else:
    gl_write_big('SRC')
  glPopMatrix()

  # draw roi button
  glPushMatrix()
  glDisable(GL_TEXTURE_2D);
  glTranslatef(-(preview_size[0]) + 2 * buttons_pos_x, (preview_size[1]) - 2 * button_scan_pos_y, 0.1)
  glColor4f(1.0, 1.0, 1.0, 0.5)
  if input_size[1] < 100:
    glBegin(GL_QUADS)
  else:
    glBegin(GL_LINE_LOOP)
  glVertex3f(-50, 50, 0.0)
  glVertex3f( 50, 50, 0.0)
  glVertex3f( 50,-50, 0.0)
  glVertex3f(-50,-50, 0.0)
  glEnd()
  glTranslatef(0, 0, 0.1)
  if input_size[1] < 100:
    glColor4f(1.0, 0.0, 0, 1.0)
    gl_write_big('ROI')
  else:
    gl_write_big('FULL')
  glPopMatrix()

  # draw a triangle
  glPushMatrix()
  glDisable(GL_TEXTURE_2D);
  glTranslatef(-(preview_size[0]), 0, 0.1)
  glColor4f(1.0, 1.0, 1.0, 1.0)
  glBegin(GL_TRIANGLES)
  glVertex3f( 0, -10, 0.0)
  glVertex3f(20,  0, 0.0)
  glVertex3f( 0, 10, 0.0)
  glEnd();
  glPopMatrix()

  glutSwapBuffers()


def key_pressed(k, x, y):
  global config
  global thread_quit
  global cam
  global input_size, output_size
  global line_index
  global process, zoom_in
  global video_thread
  global elphel

  if k == b'\x1b' or k == b'q':
    # q (quit)
    thread_quit = 1
    video_thread.join()
    # video_writer_thread.join()
    glutLeaveMainLoop()
  elif k == GLUT_KEY_RIGHT: # right (increase framertate)
    setFramerate(getFramerate() + 1)
  elif k == GLUT_KEY_LEFT: # left (decrease framertate)
    setFramerate(max(5, getFramerate() -1))
  elif k == GLUT_KEY_UP:
    if getAE():
      disableAE()
    setExposure(getExposure() * 1.025)
  elif k == GLUT_KEY_DOWN: # down (decrease exposure time)
    if getAE():
      disableAE()
    setExposure(getExposure() * 0.975)
  elif k == b'w':  # w (white balance)
    triggerWB()
  elif k == b'i': # i (input toggle)
    config["show_source"] = not config["show_source"]
  elif k == b'z':  # x (zoom)
    zoom_in = not zoom_in
  elif k == b'l':  # l (live / full frame input)
    process = False
    if config['camcontrol'] == 'ximea':
      cam.stop_acquisition()
      cam.set_offsetY(0)
      cam.set_height(full_size[1])
      cam.start_acquisition()
      cam.get_image(img)
      input_size = (img.width, img.height)
      line_index = int(input_size[1]/2) - int(config["line_height"]/2)
    elif config['camcontrol'] == 'elphel':
      elphel.setHeight(elphel.getMaxHeight())
      line_index = int(input_size[1]/2) - int(config["line_height"]/2)
      time.sleep(2)
      cam.release()
      cam = cv2.VideoCapture(config["pipeline"], cv2.CAP_GSTREAMER)
    process = True
  elif k == b'a': # a (auto exposure)e
    if getAE():
      disableAE()
    else:
      enableAE()
  elif k == b'f' or k == GLUT_KEY_F11: # f (fullscreen)
    config["fullscreen"] = not config["fullscreen"]
    if config["fullscreen"]:
      preview_size[0] = screen.width - 32
      glutFullScreen()
    else:
      preview_size[0] = preview_size[1] #int(screen.width / 2) #preview_size[1]
      if config["screenpos"] == "right":
        glutPositionWindow(screen.width - preview_size[0], 8)
      else:
        glutPositionWindow(0, 8)
      glutReshapeWindow(preview_size[0], preview_size[1])
  elif k == b'x': # x (ROI input)
    process = False
    if config['camcontrol'] == 'ximea':
      cam.stop_acquisition()
      cam.disable_auto_wb()
      cam.disable_aeag()
      cam.set_height(config["roi_height"])
      cam.set_offsetY(int(full_size[1]/2 - config["roi_height"]/2))
      cam.start_acquisition()
      cam.get_image(img)
      input_size = (img.width, img.height)
    elif config['camcontrol'] == 'elphel':
      elphel.setHeight(48)
      disableAE()
      disableAWB()
      time.sleep(2)
      cam.release()
      cam = cv2.VideoCapture(config["pipeline"], cv2.CAP_GSTREAMER)
    process = True
  elif k == b'-':
    # minus (decrease line height)
    config["line_height"] = max(1, config["line_height"] - 1)
  elif k == b'+':
    # plus (increase line height)
    config["line_height"] = config["line_height"] + 1
  elif k ==  b'h':
    # g (increase gain)
    setGain(getGain() * 1.1)
  elif k ==  b'g':
    # f (decrease gain)
    setGain(getGain() * 0.9)


def windowReshapeFunc(width, height):
  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(-width,width,-height,height);
  glMatrixMode(GL_MODELVIEW);


def gl_write(text, x, y):
  line_height = 12
  font = GLUT_BITMAP_9_BY_15
  glRasterPos2i(x, y);
  for ch in text:
    if ch == '\n':
      y = y + line_height
      glRasterPos2i(x, y)
    else:
      glutBitmapCharacter(font, ord(ch))


def gl_write_big(text):
  line_height = 12
  font = GLUT_BITMAP_9_BY_15
  glRasterPos2i(-len(text) * 9, -9);
  for ch in text:
    if ch == '\n':
      y = y + line_height
      glRasterPos2i(-len(text) * 9, y)
    else:
      glutBitmapCharacter(font, ord(ch))


def on_mouse(button, state, x, y):
  global cam
  global input_size, process
  global drag_exp, slider_exp_pos
  global drag_fps, slider_fps_pos
  global button_ae_pos_y, button_input_pos_y, button_scan_pos_y, button_wb_pos_y

  if button == GLUT_LEFT_BUTTON:
    if state == GLUT_DOWN:
      if y < 100 and abs(x - slider_exp_pos) < 20:
        drag_exp = True
      elif y > preview_size[1] - 100 and abs(x - slider_fps_pos) < 20:
        drag_fps = True
      elif abs(buttons_pos_x - x) < 50 and abs(button_ae_pos_y - y) < 50:
        if getAE():
          disableAE()
        else:
          enableAE()
      elif abs(x - buttons_pos_x) < 50 and abs(y - button_wb_pos_y) < 50:
        triggerWB()
      elif abs(x - buttons_pos_x) < 50 and abs(y - button_input_pos_y) < 50:
        config["show_source"] = not config["show_source"]
      elif abs(x - buttons_pos_x) < 50 and abs(y - button_scan_pos_y) < 50:
        # l (live / full frame input)
        process = False
        if config["camcontrol"] == "ximea":
          cam.stop_acquisition()
          if (input_size[1] == full_size[1]):
            # set ROI input
            cam.disable_auto_wb()
            cam.disable_aeag()
            cam.set_height(config["roi_height"])
            cam.set_offsetY(int(full_size[1]/2 - config["roi_height"]/2))
          else:
            # fullsize input
            cam.set_offsetY(0)
            cam.set_height(full_size[1])
          cam.start_acquisition()
          cam.get_image(img)
          input_size = (img.width, img.height)
          line_index = int(input_size[1]/2) - int(config["line_height"]/2)
          process = True
    elif state == GLUT_UP:
      drag_exp = False
      drag_fps = False


def x2microseconds(val):
  return int(((val / preview_size[0] * val / preview_size[0]) * 10000))

def microseconds2x(val):
  return min(math.sqrt(val/10000) * preview_size[0], preview_size[0])


def x2fps(val):
  return int(1 + (val / preview_size[0]) * 999)

def fps2x(val):
  return min((val - 1) / 999 * preview_size[0], preview_size[0])


def on_mouse_motion(x, y):
  global mouse_pos, last_drag_time
  global drag_exp, drag_fps

  mouse_pos[0]  =  min(max(0, x), preview_size[0])
  mouse_pos[1]  =  min(max(0, y), preview_size[1])

  #print(x2microseconds(mouse_pos[0]), microseconds2x(x2microseconds(mouse_pos[0])))
  #if (time.time() - last_drag_time > 0.1):
    #last_drag_time = time.time()
  if (drag_exp):
    if getAE():
      disableAE()
    setExposure(x2microseconds(mouse_pos[0]))

  if (drag_fps):
    setFramerate(x2fps(mouse_pos[0]))



class GpsPoller(Thread):
  def __init__(self):
    Thread.__init__(self)
    global gpsd
    global thread_quit
    gpsd = gps(mode=WATCH_ENABLE)
    self.current_value = None
    self.running = True


  def run(self):
    global gpsd
    global thread_quit
    global scanlog_file, scanlog_file_single
    global dist, output_isVideo

    last_lat = None
    last_time = None

    while not thread_quit:
      gpsd.next()
      
      if last_time != gpsd.fix.time:
        last_time = gpsd.fix.time
        if gpsd.fix.mode > 1:
          if not output_isVideo:
            write_logline(False)
          write_logline()
        if last_lat:
          rel_dist = mygeo.getDistGeod(gpsd.fix.latitude, gpsd.fix.longitude,  last_lat, last_lon)
          if rel_dist > 1:
            dist = dist + rel_dist
        last_lat = gpsd.fix.latitude
        last_lon = gpsd.fix.longitude
    scanlog_file_single.close()
    scanlog_file.close()


def write_logline(globalFile=True):
  global gpsd
  global scanlog_file, scanlog_file_single
  global total_lines_count, total_lines_index
  global dist

  target = scanlog_file
  if not globalFile:
    target = scanlog_file_single

  target.write("{}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}\n".format(
     total_lines_count if globalFile else total_lines_index,
     gpsd.fix.time,
     gpsd.fix.mode,
     gpsd.fix.latitude,
     gpsd.fix.longitude,
     gpsd.fix.altitude,
     gpsd.fix.speed,
     dist / 1000,
     gpsd.fix.track,
     gpsd.fix.climb,
     gpsd.fix.epx,
     gpsd.fix.epy,
     gpsd.fix.epv,
     len(gpsd.satellites),
     len(list(filter(lambda x: x.used, gpsd.satellites)))
  ))

"""
    old data
				utc,
				gps_status_str,
				gpsfix.latitude,
				gpsfix.longitude,
				gpsfix.altitude,
				gpsfix.speed * MPS_TO_KPH,
				distance / 1000,
				gpsfix.track,
				gpsfix.climb,
				gpsfix.epx,
				gpsfix.epy,
				gpsfix.epv,
				gpsdata->satellites_visible,
				gpsdata->satellites_used);
"""

def write_logheader(globalFile=True):
  global scanlog_file, scanlog_file_single
  target = scanlog_file
  if not globalFile:
    target = scanlog_file_single
  target.write("#line; time; status; lat; lon; alt; speed; distance; track; climb; epx; epy; epv; visible satellites; used satellites\n")


def run():
  glutInit(sys.argv)
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
  glutInitWindowSize(preview_size[0], preview_size[1])
  glutInitWindowPosition(screen.width - preview_size[0], 8)
  window = glutCreateWindow('mapylisca')
  glutDisplayFunc(draw_gl_scene)
  glutReshapeFunc(windowReshapeFunc)
  glutIdleFunc(draw_gl_scene)
  glutKeyboardFunc(key_pressed)
  glutSpecialFunc(key_pressed)
  glutMouseFunc(on_mouse)
  glutMotionFunc(on_mouse_motion)
  init_gl(preview_size[0], preview_size[1])
  if config["fullscreen"]:
    preview_size[0] = screen.width - 32
    glutFullScreen()
  else:
    preview_size[0] = preview_size[0] #int(screen.width / 2)
    if config["screenpos"] == "right":
      glutPositionWindow(screen.width - preview_size[0], 8)
    else:
      glutPositionWindow(0, 8)
  glutMainLoop()


## setings

# framerates

def getFramerate():
  global cam, config, elphel
  if config["camcontrol"] == "ximea":
    return cam.get_framerate()
  elif config["camcontrol"] == "elphel":
    return elphel.getFPS()
  else:
    return 0

def setFramerate(value):
  global cam, config, elphel
  if config["camcontrol"] == "ximea":
    cam.set_framerate(value)
  elif config["camcontrol"] == "elphel":
    elphel.setFPS(value)
  else:
    return 0

# exposure time

def getExposure():
  global cam, config, elphel
  if config["camcontrol"] == "ximea":
    return cam.get_exposure()
  elif config["camcontrol"] == "elphel":
    return elphel.getExposure()
  else:
    return 0

def setExposure(value):
  global cam, config, elphel
  if config["camcontrol"] == "ximea":
    cam.set_exposure(value)
  elif config["camcontrol"] == "elphel":
    elphel.setExposure(value)
  else:
    return 0

## auto exposre

def getAE():
  global cam, config
  if config["camcontrol"] == "ximea":
    return cam.is_aeag()
  elif config["camcontrol"] == "elphel":
    return elphel.getAutoExposureStatus()
  else:
    return 0

def enableAE():
  global cam, config, elphel
  if config["camcontrol"] == "ximea":
    cam.enable_aeag()
  elif config["camcontrol"] == "elphel":
    elphel.setAutoExposureOn()
  else:
    return 0

def disableAE():
  global cam, config, elphel
  if config["camcontrol"] == "ximea":
    cam.disable_aeag()
  elif config["camcontrol"] == "elphel":
    elphel.setAutoExposureOff()
  else:
    return 0

# auto white balane

def enableAWB():
  global cam, config, elphel
  if config["camcontrol"] == "ximea":
    cam.enable_auto_wb()
  elif config["camcontrol"] == "elphel":
    elphel.setAutoExposureOn()
  else:
    return 0

def disableAWB():
  global cam, config, elphel
  if config["camcontrol"] == "ximea":
    cam.disable_auto_wb()
  elif config["camcontrol"] == "elphel":
    elphel.setAutoWhiteBalanceOff()
  else:
    return 0

def getAWB():
  global cam, config, elphel
  if config["camcontrol"] == "ximea":
    return cam.is_auto_wb()
  elif config["camcontrol"] == "elphel":
    return elphel.getAutoWhiteBalanceStatus()
  else:
    return 0

def triggerWB():
  global cam, config, elphel
  disableAWB()
  if config["camcontrol"] == "ximea":
    if input_size[1] > 48:
      cam.set_manual_wb(1)
  elif config["camcontrol"] == "elphel":
    elphel.setAutoWhiteBalance()
  else:
    return 0

## gain

def getGain():
  global cam, config, elphel
  if config["camcontrol"] == "ximea":
    return cam.get_gain()
  elif config["camcontrol"] == "elphel":
    return elphel.getGain()
  else:
    return 0

def setGain(value):
  global cam, config, elphel
  if config["camcontrol"] == "ximea":
    cam.set_gain(value)
  elif config["camcontrol"] == "elphel":
    elphel.setGain(value)
  else:
    return 0

def quit():
  global thread_quit
  global video_thread

  thread_quit = 1
  # video_writer_thread.join()
  glutLeaveMainLoop()


if __name__ == '__main__':
  try:
    process_args()
    init()
    # start video processing thread
    video_thread = Thread(target=update_frame, args=())
    video_thread.start()

    # start output writing thread
    # video_writer_thread = Thread(target=update_write_frame, args=())
    # video_writer_thread.start()

    #start gps polling thread
    gpsd_thread = GpsPoller()
    gpsd_thread.start()

    run()

  except (KeyboardInterrupt, SystemExit): #when you press ctrl+c
    if video_thread:
      video_thread.join()

  print("Done.\nExiting.")
