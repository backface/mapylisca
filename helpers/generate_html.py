#!/usr/bin/env python3
#######################################
#
# generate a html preview files
#
# author:(c) Michael Aschauer <m AT ash.to>
# www: http:/m.ash.to
# licenced under: GPL v3
# see: http://www.gnu.org/licenses/gpl.html
#
#######################################


import glob, sys, os
import getopt, csv
from PIL import Image
from libs.myutils import *
from libs.gpxwriter import GPXWriter, GeoInfoWriter
import re

greyscale = False
input = "./"
output = "./"
name =  os.path.basename(os.getcwd())
th_height = 128
process_logs = True
process_images = True
process_html = True
verbose = True
trackfile = "track.gpx"
tracklogfile = "track.log.csv"
logstyle = "x"

def usage():
	print("""

options:
    -h, --help              print usage
    -i, --input=DIR         input path
    -o, --output=DIR        output path
    -f, --format=FORMAT     image format for cache and output    
        --name=NAME         NAME
    -n, --nologs            dont process log files
""")

def process_args():
	global input, output, format, name, process_logs
	
	try:
		opts, args = getopt.getopt(sys.argv[1:], "hi:o:nl",
			["help", "input=","output=","name=","nologs","log-only"])
	except err:
		# print help information and exit:
		print(str(err)) # will print something like "option -a not recognized"
		usage()
		sys.exit(2)

	for o, a in opts:
		if o == "-v":
			verbose = True
		elif o in ("-h", "--help"):
			usage()
			sys.exit()
		elif o in ("-o", "--output"):
			output = a
		elif o in ("-i", "--input"):
			input=a
		elif o in ("-f", "--format"):
			format = a
		elif o in ("--name"):			
			name = a
		elif o in ("-n","--nologs"):
			process_logs = False
		elif o in ("-l","--logs-only"):
			process_logs = True
			process_images = False
			process_html = False			
		else:
			assert False, "unhandled option"
		

if __name__ == "__main__":

	process_args()

	html_file = output + "/index.html"
	thumb_path = output + "/%d/" % (th_height)
	meta_file = "%s.txt" % output.split(".")[0]	
		
	count = 0
	leng = 0

	files = glob.glob(input + "/*.jpg")
	files.sort()
	
	if files == []:
		print("no files found for", input)
		exit()

	source = ""
	info = ""

	if process_logs:
		infowriter = GeoInfoWriter(output + "/info.txt")
		gpxwriter = GPXWriter(output + "/" + trackfile)
		logwriter = csv.writer(open(output + "/" + tracklogfile, "w"),delimiter=";")
	else:
		print("don't process logs")
	
	for file in files:

		thumb_file = "%s/%s" % (thumb_path,os.path.basename(file))
		log_file = "%s.log" % file

		if process_images:
			img = Image.open(file)
			w = img.size[0]
			h = img.size[1]
			ratio = th_height / float(h)
			th_width = int(w * ratio)

		if process_logs:

			if logstyle == "one":
				if count == 0:
					logreader = csv.reader(open(file + ".log","r"), delimiter=" ")
					for line in logreader:
						if not line[0] == ";": #temp hack
							logwriter.writerow(line)
							#print float(line[2]), float(line[3])
							try:
								infowriter.addPoint(float(line[3]), float(line[4]))
								gpxwriter.addTrackpoint(float(line[3]), float(line[4]))
							except:
								print("value error")
							
			else:		
				if os.path.exists(file + ".log"):
					logreader = csv.reader(open(file + ".log","r"), delimiter=";")
					for line in logreader:
						logwriter.writerow(line)
						infowriter.addPoint(
                float(line[3]), float(line[4]),
                line[1], float(line[5]), float(line[6]))
						gpxwriter.addTrackpoint(float(line[3]), float(line[4]),
                line[1], float(line[5]), float(line[6]),
                int(line[2]),"",line[0]
              )
								
		if not os.path.exists(thumb_file) and process_images:
			# generate thumbs
			if verbose:
				print("make thumnail from %s" % os.path.basename(file)	)		
			img_out = img.resize((th_width,th_height),Image.ANTIALIAS)
			createPath(thumb_file)
			img_out.save(thumb_file,quality=95)
						
		if process_html:
			source += '<a href="%s"><img border="0" src="%s" alt="" height="%d" width="%d" /></a>\n' % \
			(file, thumb_file, th_height, th_width)

		count+=1

	if process_logs: 
		info = infowriter.getInfoStringHTML()
		infowriter.save()
		gpxwriter.save()
		info += '&raquo; <a href="%s">trackfile</a>' % trackfile
	else:
		info = "no gps log data available."
		
	# now write to files	
	print("%d files found." % count)
	if process_html:
		print("generating html..")

		# write html
		f = open(html_file, 'w')	
		f.write('''<html>
		<head>
			<title>%s</title>
			<meta http-equiv="content-type" content="text/html; charset=UTF-8" />
			<style type="text/css" media="screen">
				body {font-family: Georgia, Times New Roman, Times, serif}
				a {padding:0px;margin:0px}
				img {padding:0px;margin:0px;margin-right:-4px;margin-top:5px}
			</style>
		<head>
		<body>
		<h2>%s</h2>
		<div id="thumbs">
			%s
		</div>
		</html>''' % ( name, name, source  )
		)
		f.close()
