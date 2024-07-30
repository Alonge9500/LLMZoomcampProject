docker run -it \
    --rm \
    --name elasticsearch \
    -m 4GB \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3

docker run -it \
    -v ollama:/root/.ollama \
    -p 11434:11434 \
    --name ollama \
    ollama/ollama

docker exec -it ollama bash
ollama pull phi3


curl -fsSL https://ollama.com/install.sh | sh

ollama start
ollama pull phi3
ollama run phi3

### Running Ngrok on docker
* This is use to forwarad elastic search port for usability in sturn cloud
docker run --net=host -it -e NGROK_AUTHTOKEN=2jxmLDs217VQopAUJgMQlmlBNdR_4LuHhU4o7UqQ1sGCbt958 ngrok/ngrok:latest http 9200