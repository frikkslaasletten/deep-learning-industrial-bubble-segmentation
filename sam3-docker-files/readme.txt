# First time
docker pull 676638/sam3
docker create -it --gpus all 676638/sam3
docker ps -a
docker start -ai CONTAINER_NAME
conda init
exit
docker start -ai CONTAINER_NAME
conda activate sam3
export HF_TOKEN=""
python main.py

# Subsequent times
docker start -ai CONTAINER_NAME
python main.py

