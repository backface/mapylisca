gst-launch-1.0 rtspsrc location=rtsp://192.168.0.9:554 latency=100 protocols=0x00000001 ! rtpjpegdepay! jpegdec ! autovideosink
