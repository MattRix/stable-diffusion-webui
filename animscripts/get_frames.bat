rmdir "frames" /s /q
mkdir "frames"
::actual headphones frame rate is 25
ffmpeg -ss 6 -i "Thinking-82130.mp4" -t 10 -s 960x512 -r 12 "in_images_thinking/%%04d.png" 
