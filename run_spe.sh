set -e
set -x
make llama-speculative
#make LLAMA_SERVER_SSL=true llama-server
./llama-speculative  -m ../gguf/qwen2-0_5b-instruct-q3_k_m.gguf -md ../gguf/qwen2-0_5b-instruct-q3_k_m.gguf -p "# Dijkstra's shortest path algorithm in Python (4 spaces indentation) + complexity analysis:\n\n" \
-e -ngl 999 -ngld 999 -t 4 -n 512 -c 4096 -s 21 --draft 16 -np 1 --temp 0.0
#./llama-server -m ../gguf/qwen2-0_5b-instruct-q3_k_m.gguf -c 2048  --all-logits
