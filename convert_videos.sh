find /home/haifengh/.cache/huggingface/cvrr-es/CVRR-ES -name '*.webm' -exec bash -c '
for f in "$@"; do
  out="${f%.webm}.mp4"
  printf "Converting: <%s>\n" "$f"
  ffmpeg -y -i "$f" \
    -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
    -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
    "$out"
done
' bash {} +