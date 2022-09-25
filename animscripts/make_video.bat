
::note how we use %% because it's a bat file so it needs to be escaped
::crf is quality, usually 5-10 is good (lower is better), r is framerate, s is size
::ffmpeg -r 30 -f image2 -s 512x512 -i "out_images/%%04d.png" "results/gem_out.gif"

@echo off
set /p FOLDERID="Folder id: "

::regular mp4
ffmpeg -r 25 -f image2 -s 640x384 -i "out_images_%FOLDERID%/%%04d.png" -vcodec libx264 -crf 2  -pix_fmt yuv420p "results/thinking_%FOLDERID%.mp4"

::shareable size gif
ffmpeg -r 25 -f image2 -s 640x384 -i "out_images_%FOLDERID%/%%04d.png" -vf "fps=25,scale=640:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:reserve_transparent=0[p];[s1][p]paletteuse" "results/thinking_%FOLDERID%.gif"
