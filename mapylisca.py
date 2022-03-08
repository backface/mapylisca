#!/usr/bin/env python
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
from PIL import ImageDraw, Image
from ctypes import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from threading import Thread
from gps import gps
#import cv2
import time, datetime
import sys, os
import numpy as np
import screeninfo
import imageio
import copy

output_path = "scan-data/"
file_format = "JPEG"
line_height = 12
roi_height = 16
output_size = (2064, 512)
input_size = (640, 480)
offset = 16

display = True
write_log_file = False
show_source = False
process = True
fullscreen = False

line_count = 1
line_index = int(input_size[1]/2) - int(line_height/2)
exposure = 5000

cam = None
img = None
data = None
scan_data = None
black_frame = None
last_full_frame = None
texture_id = 0

time_start = 0
elapsed = 0
elapsed_total = 0
fps_timer_start = 0
fps = 0
text = ''

thread_quit = 0
videowriter = None
video_thread = None
tile_buffer = []
tile_texture_ids = []
shift_tiles = True

screen_id = 0
screen = screeninfo.get_monitors()[screen_id]
#width, height = screen.width, screen.height
preview_size = (screen.height - 32, screen.height - 32)


def init():
  global cam, img, process
  global data, scan_data
  global input_size, full_size, output_size
  global line_height, fps, fps_timer_start, time_start, elapsed
  global videowriter  
  global video_thread
  global tile_buffer
  global black_frame
  global last_full_frame
    
  # init camera
  cam = xiapi.Camera()
  print('Opening first camera...')
  cam.open_device()
  
  # settings
  cam.set_imgdataformat('XI_RGB24')
  cam.set_exposure(exposure)
  cam.set_framerate(100)
  #cam.enable_auto_wb()
  #cam.enable_aeag()
  cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE_LIMIT")
  img = xiapi.Image()
  print('Starting data acquisition...')
  cam.start_acquisition()
  
  # get first image and dimensions
  cam.get_image(img)
  input_size = (img.width, img.height)
  full_size = input_size
  output_size = (img.width, output_size[1])
  black_frame = np.zeros((output_size[1], output_size[0], 3), dtype=c_ubyte)
  scan_data = copy.deepcopy(black_frame)
  last_full_frame = copy.deepcopy(black_frame)
  

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
  print("get_temp :", cam.get_temp())
  print("get_chip_temp :", cam.get_chip_temp())
  print("get_framerate :", cam.get_framerate())
  print("get_acq_timing_mode", cam.get_acq_timing_mode())
  print("get_limit_bandwidth_mode",  cam.get_limit_bandwidth_mode())
  print("get_offsetY_increment", cam.get_offsetY_increment())
  print("=========================")
  
  time_start = time.time()
  fps_timer_start =  time.time()

  outfile = '{}/{}.avi'.format(output_path, time.strftime("%Y-%m-%d_%H-%M_%S", time.gmtime()))
  os.makedirs(os.path.dirname(outfile), exist_ok=True)
  videowriter = imageio.get_writer(outfile, 
    format='FFMPEG', 
    mode='I', 
    fps=25, 
    codec='mjpeg', 
    output_params=['-q:v','1'],
    #pixelformat='yuvj420p'
  )
  
  video_thread = Thread(target=update_frame, args=())
  video_thread.start()
  
  
  #fmpeg_log_level="info",
  #output_params=['-x265-params','keyint=50'])                                      
  #pixelformat='vaapi_vld')
  
     
def init_gl(width, height):
  global texture_id

  glClearColor(0,0,0, 1.0)
  glClearDepth(1.0)
  glDepthFunc(GL_LESS)
  #glDepthFunc(GL_LEQUAL);
  glEnable(GL_DEPTH_TEST)
  #glDisable(GL_CULL_FACE);
  #glShadeModel(GL_SMOOTH)
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)
  glMatrixMode(GL_MODELVIEW)
  glEnable(GL_TEXTURE_2D)
  # create texture
	#GL_LINEAR is better looking than GL_NEAREST but seems slower.. ?
  #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
  #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  texture_id = glGenTextures(1)


def update_frame():
  global cam, img, text
  global data, scan_data
  global input_size, full_size, output_size
  global line_height, fps, fps_timer_start, elapsed, line_count
  global videoout
  global shift_tiles
  global blackframe
  global last_full_frame
  
  line_input_index = int(input_size[1]/2) - int(line_height/2)
  line_output_index = 0
  elapsed_total = 0
  frame_count = 0
  
  while(True):
    if process:
      start_frame = time.time()
      
      cam.get_image(img)
      data = img.get_image_data_numpy()
      line_input_index = int(input_size[1]/2) - int(line_height/2)
      
      for i in range(0, line_height):        
        scan_data[line_output_index,:] = data[line_input_index + i,:]
        
      #for i in range(0, line_height):
      #  scan_data = np.append(scan_data, [data[line_index + i,:]], 0)
      #  scan_data = np.delete(scan_data, 0, 0)
      
        line_output_index = line_output_index + 1
        if line_output_index >= output_size[1] -1:
          videowriter.append_data(scan_data)
          last_full_frame = copy.deepcopy(scan_data)
          shift_tiles = True
          scan_data = black_frame.copy()
          line_output_index = 0
          frame_count = frame_count + 1

      line_count = line_count + 1       
      
      if (line_count % 25 == 0):
        end = time.time()
        seconds = end - fps_timer_start
        fps  = 25 / seconds
        fps_timer_start = end
      
      text = '\r{:02.0f}:{:02.0f}:{:02.0f} | LH={:0.0f} | #{:0.0f}/#{:0.0f} | REQ: {:02.0f}fps / REAL: {:02.0f}fps | EXP: {:2.2f}ms ({:0.0f}dB) | FRAME: {:3.0f}ms '.format(
        (elapsed_total/3600.0),  (elapsed_total/60) % 60, (elapsed_total % 60), 
        line_height,
        frame_count,
        line_count,        
        cam.get_framerate(), 
        fps, 
        cam.get_exposure()/1000,
        cam.get_gain(),
        elapsed * 1000
      )
      print(text, end=" ... ")      
      elapsed = time.time() - start_frame
      elapsed_total = time.time() - time_start   
  
    if thread_quit:
      break
  
  print("")
  cam.stop_acquisition
  cam.close_device()
  videowriter.close()


def draw_gl_scene():
  global texture_id, text, offset
  global line_count
  global tile_buffer
  global tile_texture_ids
  global shift_tiles
  global last_full_frame
  global black_frame
  global scan_data
  
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
  glLoadIdentity()

  ratio = 1
  if (show_source):
    frame = data
    ratio = input_size[1]/input_size[0]
  else:
    frame = scan_data
    ratio = preview_size[0]/output_size[0]
    tile_size = (output_size[0] * ratio, output_size[1] * ratio)

  # prepare scan texture

  #tx_image = cv2.flip(frame, 0)
  tx_image = Image.fromarray(frame)
  ix = tx_image.size[0]
  iy = tx_image.size[1]
  tx_image = tx_image.tobytes('raw', 'BGRX', 0, -1)
  
  glEnable(GL_TEXTURE_2D);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, tx_image)
  glBindTexture(GL_TEXTURE_2D, texture_id)   

  # setup tile textures
  if len(tile_texture_ids) == 0:
    print('setup tiles')
    num_tiles = int(preview_size[1] / (output_size[1] * preview_size[0]/output_size[0]))
    tile_texture_ids = glGenTextures(num_tiles)
    black = Image.fromarray(black_frame).tobytes('raw', 'BGRX', 0, -1)
    for i in range(0, num_tiles):
      tile_buffer.append(black)
      glBindTexture(GL_TEXTURE_2D, tile_texture_ids[i])   
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); 
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE,  tile_buffer[i] )
  
  # shift tiles
  if shift_tiles:
    for i, buffer_id in enumerate(tile_texture_ids):
      if i < len(tile_texture_ids) - 1:
        tile_buffer[i] = copy.deepcopy(tile_buffer[i+1]) 
      else:
        tile_buffer[i] = Image.fromarray(last_full_frame).tobytes('raw', 'BGRX', 0, -1)
      glBindTexture(GL_TEXTURE_2D, buffer_id)   
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); 
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE,  tile_buffer[i] )
    shift_tiles = False
 
  # show input souce
  if (show_source):  
    offset = 0    
    glPushMatrix()
    glTranslatef(0.0, 0.0, 0.0);
    glRotatef(90, 0, 0, 1); 
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex3f(-preview_size[0], -preview_size[1] * ratio - offset, 0);
    glTexCoord2f(1, 0); glVertex3f( preview_size[0], -preview_size[1] * ratio - offset, 0);
    glTexCoord2f(1, 1); glVertex3f( preview_size[0],  preview_size[1] * ratio - offset, 0);
    glTexCoord2f(0, 1); glVertex3f(-preview_size[0],  preview_size[1] * ratio - offset, 0);
    glEnd();
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture_id) 
    glPopMatrix()
  
  else:
    #offset = (line_count % output_size[1]) * line_height * ratio
    offset = 0 
    # tile buffer
    for i, buffer_id in enumerate(tile_texture_ids):
      glPushMatrix()
      glTranslatef(-offset + preview_size[1] - ((len(tile_texture_ids) - i + 1) * 2 * tile_size[1] - tile_size[1]), 0.0, 0.0);
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
  
    # main scan frame
    glPushMatrix()
    glTranslatef(0.0, 0.0, 0.0);
    glTranslatef(-offset + preview_size[1] - tile_size[1], 0.0, 0.0);  
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
    
  # draw a line
  glPushMatrix()
  glDisable(GL_TEXTURE_2D);
  glTranslatef(0.0, 0.0, -1)
  glColor4f(1.0, 1.0, 1.0, 1.0)
  glBegin(GL_LINES)
  glVertex3f(-preview_size[1], 0.0, 0.0);
  glVertex3f( preview_size[1], 0.0, 0.0);
  glEnd();
  glPopMatrix()

  # draw text info
  glPushMatrix()
  glTranslatef(0.0, 0.0, -0.1)
  gl_write(text, 32, 32)
  glDisable(GL_TEXTURE_2D);
  glPopMatrix()
  
  glutSwapBuffers()


def key_pressed(k, x, y):
  global thread_quit
  global cam
  global input_size, output_size
  global line_index, line_height
  global show_source, process, fullscreen
  global video_thread

  # q(quit)
  if k == b'\x1b' or k == b'q':
    thread_quit = 1
    video_thread.join()
    glutLeaveMainLoop()
  elif k == GLUT_KEY_RIGHT:
    cam.set_framerate(cam.get_framerate() + 1)
  elif k == GLUT_KEY_LEFT:
    cam.set_framerate(max(5, cam.get_framerate() - 1))
  elif k == GLUT_KEY_UP:
    cam.set_exposure(int(cam.get_exposure() * 1.1))
  elif k == GLUT_KEY_DOWN: 
    cam.set_exposure(int(cam.get_exposure() * 0.9))
  # w (white balance)
  elif k == b'w':             
    cam.set_manual_wb(1)
   # i (input toggle)
  elif k == b'i':            
    show_source = not show_source
  # a (all / full frame input)
  elif k == b'a':             
    process = False
    cam.stop_acquisition()
    cam.set_offsetY(0)
    cam.set_height(full_size[1])
    cam.start_acquisition()     
    cam.get_image(img)
    input_size = (img.width, img.height)
    line_index = int(input_size[1]/2) - int(line_height/2)
    process = True
  # f (fullscreen)
  elif k == b'f': 
    fullscreen = not fullscreen
    if fullscreen:
      glutFullScreen()
    else:
      glutPositionWindow(screen.width - preview_size[0], 8)
      glutReshapeWindow(preview_size[0], preview_size[1])
  # x (ROI input)
  elif k == b'x': 
    process = False
    cam.stop_acquisition()
    cam.set_height(roi_height)
    cam.set_offsetY(int(full_size[1]/2))
    cam.start_acquisition()
    cam.get_image(img)
    input_size = (img.width, img.height)
    process = True
  elif k == b'-':
    line_height = max(1, line_height - 1)
  elif k == b'+':
    line_height = line_height + 1
  elif k ==  b'g':
    cam.set_gain(int(cam.get_gain() + 1))
  elif k ==  b'f': 
    cam.set_gain(int(cam.get_gain() * 0.9))


def windowReshapeFunc(width, height):
  glViewport(0, 0, width, height); 
  glMatrixMode(GL_PROJECTION); 
  glLoadIdentity();
  gluOrtho2D(-width,width,-height,height);
  glMatrixMode(GL_MODELVIEW);


def gl_write(text, x, y):
  line_height = 12
  font = GLUT_BITMAP_HELVETICA_12
  glColor4f(1.0, 1.0, 1.0, 1.0);
  glRasterPos2i(x, y);
  for ch in text:
    if ch == '\n':
      y = y + line_height
      glRasterPos2i(x, y)
    else:
      glutBitmapCharacter(font, ord(ch))

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
  init_gl(preview_size[0], preview_size[1])
  glutMainLoop()


if __name__ == '__main__':
  init()
  run()
