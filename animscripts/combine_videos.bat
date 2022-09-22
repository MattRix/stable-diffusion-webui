
::speed up the framerate. the -r must come before the inputs in the args list for some reason
ffmpeg -r 40 -f concat -safe 0 -i giflist.txt -vf "loop=0,fps=40,scale=640:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:reserve_transparent=0[p];[s1][p]paletteuse" results/headphones_COMBINED.gif
::ffmpeg -i results/headphones_COMBINED.gif -loop 0 results/headphones_COMBINED.gif
::failed attempts below. I think no matter what it doesn't see the gifs as the same format, so we have to use the mp4s instead
::ffmpeg -r 25 -f image2 -s 640x384 -i "results/headphones_watercolor_out.gif" -i "results/headphones_out_black_man.gif" -vf "fps=25,scale=640:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:reserve_transparent=0[p];[s1][p]paletteuse" "results/headphones_COMBINED.gif
::ffmpeg  -i "results/headphones_watercolor.gif" -i "results/headphones_black_man.gif" "results/headphones_COMBINED.gif
::ffmpeg -i "concat:results/headphones_watercolor.gif|results/headphones_black_man.gif" -c copy "results/headphones_COMBINED.gif
::ffmpeg -f concat -safe 0 -i giflist.txt -c copy COMBINED.mp4
::ff::mpeg -i results/headphones_watercolor.mp4 temp1.mp4
::ffmpeg -i results/headphones_black_man.mp4 temp2.mp4
::ffmpeg -i "concat:temp1.mp4|temp2.mp4" output.mp4
