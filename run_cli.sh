set -e
set -x
make llama-cli
#make LLAMA_SERVER_SSL=true llama-server
./llama-cli  -m ../gguf/qwen2-0_5b-instruct-q3_k_m.gguf -p 'hello, who are you' --predict 30 -c 1
#./llama-server -m ../gguf/qwen2-0_5b-instruct-q3_k_m.gguf -c 2048  --all-logits
