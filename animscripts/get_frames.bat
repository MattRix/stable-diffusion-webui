rmdir "frames" /s /q
mkdir "frames"
::actual headphones frame rate is 25
ffmpeg -ss 10 -i "headphones.mp4" -t 5 -s 960x512 -r 25 "in_images/%%04d.png" 
