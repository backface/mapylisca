#!/usr/bin/python

from PIL import Image, ImageDraw
import glob, os, sys, getopt, string, math
#import geoname

def usage():
	print("usage")
	print("")
	print("makedunavtiles.py -i [directory] -o [filename.ext]\n")
	print("more options:")
	print("  -v                           be verbose")
	print("  -h | --help                  show this screen")
	print("  -s | --size WIDTHxHEIGHT     size of tiles generated")
	print("  -x | --width WIDTH           width of tiles generated")
	print("  -y | --height HEIGHT         height of tiles generated")
	print("  -l | --logs                  process log files also")
	print("  -n | --logs-only             just process log files - no images")
	print("  -d | --dpi DPI               DPI setting of generated images")
	print("  -p | --offset INT            offest by INT pixels ??")
	print("  -m | --log-images            build log-file images")
	print("")

def main():
	try:
		opts, args = getopt.getopt(sys.argv[1:], "i:o:x:y:s:d:vlp:na:m", ["help", "output=","input=","height=","width=","dpi=","process_logs=","size=","offset=","aspect=","log-images"])
	except getopt.GetoptError:
		# print help information and exit:
		usage()
		sys.exit(2)

	dpi_info = (300,300)
	output_files = "output.tif"
	input_files = "*.jpg"
	out_w = 0
	out_h = 0
	verbose = True
	processLogs = False
	has_log_file = False
	offset = 0
	saveImages = True
	doLogImages = False
	aspect = 1
	summary = True

	for o, a in opts:
		if o == "-v":
			verbose = True
		if o in ("-h","--help"):
			usage()
			sys.exit()
		if o in ("-p","--offset"):
			offset = int(a)
		if o in ("-o", "--output"):
			output_files = a
		if o in ("-i", "--input"):
			input_files = a
		if o in ("-l", "--logs"):
			processLogs = True
		if o in ("-x", "--width"):
			out_w = int(a)
		if o in ("-y", "--height"):
			out_h = int(a)
		if o in ("-d", "--dpi"):
			dpi_info = (int(a),int(a))
		if o in ("-s", "--size"):
			a = a.split("x")
			out_w = int(a[0])
			out_h = int(a[1])
		if o in ("-n","--logs-only"):
			saveImages = False
		if o in ("-m","--log-image"):
			doLogImages = True
		if o in ("-a","--aspect"):
			aspect = float(a)
		

	x = 0
	y = 0
	distance = 0
	dd_distance = 0
	total_distance = 0
	in_log_lines = []
	out_log_lines = []
	scale_factor = aspect
	dist = 0
	out_w = int (out_w * aspect)

	out_file, out_ext = os.path.splitext(output_files)
		
	inlist = glob.glob(input_files);
	inlist.sort()

	i = 0

	if offset > 0:
		del inlist[0:offset-1]
		
	for infile in inlist:	
		file, ext = os.path.splitext(infile)
		im = Image.open(infile)
		in_w, in_h = im.size	
    
		if out_w == 0:
			out_w = in_h
    
		if out_h == 0:
			out_h = in_h

		if verbose:
			print("read: ", infile) # im.size, im.mode, im.info)

		if in_h != out_h:
			scale_factor = float(out_h)/float(in_h)
			if saveImages:
				im = im.resize( (int(in_w * (scale_factor)), out_h), Image.ANTIALIAS)
			in_w, in_h = im.size
			if verbose:
				print("resize: ", infile, im.size)

		#if aspect != 1:
		#	if saveImages:
		#		im = im.resize( (int(in_w * (aspect)), out_h), Image.ANTIALIAS)
		#	in_w, in_h = im.size
		#	if verbose:
		#		print("resize: ", infile, im.size

		if processLogs:
			logfile = file + ".log"
			try:
				f = open(logfile)
			except IOError:
				print("err: could not open log file")
				has_log_file = False
			else:
				in_log_lines = []
				for line in f.readlines():
					in_log_lines.append(line.split(" "))
				f.close()
				has_log_file = True	
					
		if x == 0:
			if saveImages:
				out_img = Image.new("RGB",(out_w, out_h),(255,255,255))
				out_img.info['dpi'] = dpi_info
			if doLogImages:
				out_log_img = Image.new("RGB",(out_w, 10),(255,255,255))
				out_log_img.info['dpi'] = dpi_info				
			i += 1
			out_log_lines = []

		if doLogImages & has_log_file:
			
			if distance == 0: 
				src_log_line = in_log_lines[0]
			
			for line in in_log_lines:
				dist = getDistance(src_log_line[2], src_log_line[3], line[2], line[3])		
				distance += dist		
				total_distance +=dist
				if distance >= 1:
					draw = ImageDraw.Draw(out_log_img)
					draw.line((x + int(line[0])*scale_factor, 0, x + int(line[0])*scale_factor, im.size[1]), fill=0, width=2)
					del draw 
					src_log_line = line
					#places = geoname.findNearbyPlaceName(line[2],line[3])
					#for b in places.geoname:
				    #	print line[2],line[3], b.name, b.distance
					distance = 0	
					
		if has_log_file & (not doLogImages):
			
			first = in_log_lines[0]
			last = in_log_lines[len(in_log_lines)-1]
			
			dist = getDistance(first[2], first[3], last[2], last[3])					
			total_distance +=dist
			dd_distance += dist	

		if x + in_w <= out_w:
			
			if saveImages:
				out_img.paste( im.copy(), (x, 0, x + in_w, out_h) )
			
			if processLogs & has_log_file:
				for line in in_log_lines:
					line[0] = str(int(line[0])*scale_factor + x)
					out_log_lines.append(string.join(line," "))				
			
			x = x + in_w	
		
		else:
			
			if saveImages:
				out_img.paste( im.crop( (0, 0, out_w - x, in_h) ), (x, 0, out_w, out_h) )
			
			if processLogs & has_log_file:
				for line in in_log_lines:
					if (int(line[0])*scale_factor + x) < out_w:
						line[0] = str(int(line[0])*scale_factor + x)						
						out_log_lines.append(string.join(line," "))
						in_log_lines.remove(line)
				f = open("%s-%05d%s" % (out_file,i,".log"),"w")
				f.writelines(out_log_lines)
				f.close			
		
			if saveImages:
				if verbose:
					print("save: ", "%s-%05d%s" % (out_file, i, out_ext)) #, out_img.size, out_img.mode, out_img.info)
				os.makedirs(os.path.dirname("%s-%05d%s" % (out_file, i, out_ext)), exist_ok=True)
				out_img.save("%s-%05d%s" % (out_file, i, out_ext), dpi=dpi_info, quality=100)
				out_img = Image.new("RGB",(out_w, out_h),(255,255,255) )
				out_img.info['dpi'] = dpi_info
				out_img.paste( im.crop( (out_w - x, 0, in_w, in_h)), (0, 0, in_w - (out_w - x), out_h) )
				
			if doLogImages & processLogs:
				os.makedirs(os.path.dirname("%s-%05d.log%s" % (out_file,i,out_ext)), exist_ok=True)
				out_log_img.save("%s-%05d.log%s" % (out_file,i,out_ext), dpi=dpi_info)
				out_log_img = Image.new("RGB",(out_w, 10),(255,255,255))
		
			if processLogs & has_log_file:
				if verbose:
					print("distance: ", "%0.2f" % (total_distance))
				out_log_lines = []
				for line in in_log_lines:
					if int(line[0])*scale_factor + x > out_w: 
						line[0] = str( int(line[0])*scale_factor + x - out_w )
						if line[0] > 0:
							out_log_lines.append(string.join(line," "))
				if summary:
					ffline = "distance:  %0.2f km\n" % (dd_distance)
					ff = open("%s-%05d%s" % (out_file,i,".txt"),"w")
					ff.write(ffline)
					ff.close
					dd_distance = 0

			in_log_lines = []			
			x =  in_w - (out_w - x)
			i += 1

		if x == out_w:		
		
			if saveImages:
				if verbose:
					print("save: ", "%s-%05d%s" % (out_file,i,out_ext)) #, out_img.size, out_img.mode, out_img.info )			
				out_img.save( "%s-%05d%s" % (out_file,i,out_ext), dpi=dpi_info, quality=100 )
		
			if doLogImages & processLogs:
				out_log_img.save( "%s-%05d.log%s" % (out_file,i,out_ext), dpi=dpi_info )				
		
			if processLogs:
				f = open("%s-%05d%s" % (out_file,i,".log"),"w")
				f.writelines(out_log_lines)
				f.close

				if summary:
					ffline = "distance:  %0.2f km\n" % (dd_distance)
					ff = open("%s-%05d%s" % (out_file,i,".txt"),"w")
					ff.write(ffline)
					ff.close
					dd_distance = 0
		
				if verbose:
					print("distance: ", "%0.2f" % (total_distance))
			x = 0		

	if x!= 0:
		
		if saveImages:
			if verbose:
				print("save: ", "%s-%05d%s" % (out_file,i,out_ext)) #, out_img.size, out_img.mode, out_img.info)
			out_img.save( "%s-%05d%s" % (out_file,i,out_ext),  dpi=dpi_info, quality=100)
		
		if doLogImages:
			out_log_img.save( "%s-%05d.log%s" % (out_file,i,out_ext), dpi=dpi_info )
		
		if processLogs:
			if verbose:
				print("distance: ", "%0.2f" % (total_distance))

			f = open("%s-%05d%s" % (out_file,i,".log"),"w")
			f.writelines(out_log_lines)
			f.close

			if summary:
				ffline = "distance:  %0.2f km\n" % (dd_distance)
				ff = open("%s-%05d%s" % (out_file,i,".txt"),"w")
				ff.write(ffline)
				ff.close
				dd_distance = 0	
		
			
	print("generated files in %s" % out_file)
	print("number of files: %d" % i)
	print("TOTAL distance: %0.2f km" % total_distance)
	

def getDistance(x1,y1,x2,y2):
	PI = math.pi
	lon1  =  float(x1) * PI/180  
	lon2  =  float(x2) * PI/180  
	lat1  =  float(y1) * PI/180
	lat2  =  float(y2) * PI/180
	theta  = lon1 - lon2	
	if theta != 0:
		dist =  6371.2 * math.acos( math.sin(lat1)  *  math.sin(lat2) +  math.cos(lat1)  *  math.cos(lat2)  *  math.cos(theta)	)
		return dist
	else:
		return 0



if __name__ == "__main__":
	main()


