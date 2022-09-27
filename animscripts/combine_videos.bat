
ffmpeg -r 25 -f concat -safe 0 -i giflist.txt -vf "loop=0,fps=25,scale=640:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:reserve_transparent=0[p];[s1][p]paletteuse" results/thinking_COMBINED.gif
