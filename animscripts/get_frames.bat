::rmdir "input/girl" /s /q
::mkdir "input/girl"

::ffmpeg -ss 8 -i "Girl - 46949.mp4" -t 4 -s 960x512 -r 24 "input/girl/%%04d.png" 

::ffmpeg -ss 8 -i "Girl - 46949.mp4" -t 4 -s 768x384 -r 24 "input/girl/%%04d.png" 

::skyscrapers source is 60fps 1280x720 (1.7777)
ffmpeg -ss 0 -i "Skyscrapers - 91744.mp4" -t 17 -s 1280x720 -r 6 "input/skyscrapers/%%04d.png" 
