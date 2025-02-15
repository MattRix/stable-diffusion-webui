
::note how we use %% because it's a bat file so it needs to be escaped
::crf is quality, usually 5-10 is good (lower is better), r is framerate, s is size
::ffmpeg -r 30 -f image2 -s 512x512 -i "out_images/%%04d.png" "results/gem_out.gif"

@echo off
set /p FOLDERID="Folder id: "

::FPS 20 and SKIP 2 to make it play at half speed, for example
set WIDTH=1152
set HEIGHT=640
set FPS=10
set SKIP=1
set TRUEFPS=%FPS%/%SKIP%

ffmpeg -r %TRUEFPS% -f image2 -s %WIDTH%x%HEIGHT% -i "output/%FOLDERID%/%%04d.png" -vf "select='not(mod(n,%SKIP%))',setpts=N/12/TB" -vcodec libx264 -crf 2  -pix_fmt yuv420p "results/%FOLDERID%_halftime.mp4"

::shareable size gif
ffmpeg -r %TRUEFPS% -f image2 -s %WIDTH%x%HEIGHT% -i "output/%FOLDERID%/%%04d.png" -vf "select='not(mod(n,%SKIP%))',setpts=N/12/TB,scale=%WIDTH%:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:reserve_transparent=0[p];[s1][p]paletteuse" "results/%FOLDERID%_halftime.gif"
