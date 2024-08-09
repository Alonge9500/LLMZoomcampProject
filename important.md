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

### Running Elastic Search on Saturn Cloud 
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.9.1-linux-x86_64.tar.gz
tar -xzf elasticsearch-8.9.1-linux-x86_64.tar.gz
cd elasticsearch-8.9.1

If you experience space error
cd /tmp and install elasticsearch here

To run 
./bin/elasticsearch

To Test
curl http://localhost:9200

If you experience security error in connecting to elastic search

Enter your elastic.yml file and chanege the following from true to false

xpack.security.enabled: false
xpack.security.transport.ssl.enabled: false
xpack.security.http.ssl.enabled: false


You can open it with nano
To install Nano
sudo apt install nano

Sudo nano elasticsearch/elasticsearch.yml

Edit the above mention and save your file then restart elastic search


wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
sudo tar xvzf ./ngrok-v3-stable-linux-amd64.tgz -C /usr/local/bin
ngrok config add-authtoken 2jxmLDs217VQopAUJgMQlmlBNdR_4LuHhU4o7UqQ1sGCbt958


Top 20 highest files
du -ah /home/jovyan | sort -rh | head -n 20
