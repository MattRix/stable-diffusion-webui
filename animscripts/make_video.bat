
::note how we use %% because it's a bat file so it needs to be escaped
::crf is quality, usually 5-10 is good (lower is better), r is framerate, s is size
::ffmpeg -r 30 -f image2 -s 512x512 -i "out_images/%%04d.png" "results/gem_out.gif"

@echo off
set /p FOLDERID="Folder id: "

set WIDTH=1152
set HEIGHT=640
set FPS=20

::regular mp4
ffmpeg -r %FPS% -f image2 -s %WIDTH%x%HEIGHT% -i "output/%FOLDERID%/%%04d.png" -vcodec libx264 -crf 2  -pix_fmt yuv420p "results/%FOLDERID%.mp4"

::shareable size gif
ffmpeg -r %FPS% -f image2 -s %WIDTH%x%HEIGHT% -i "output/%FOLDERID%/%%04d.png" -vf "fps=%FPS%,scale=%WIDTH%:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:reserve_transparent=0[p];[s1][p]paletteuse" "results/%FOLDERID%.gif"
