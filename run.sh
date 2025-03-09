# It is recommended to create a shared cache for both the
# host and the docker. Eg. designate a folder on your
# machine and set it to be the huggingface cache. Then,
# bind the directory into docker and set it to be also
# the cache there.
#
# Eg.
# (host)   : export HF_HOME=/mnt/whatever
# --mount --mount "type=bind,src=/mnt/whatever,dst=/path/to/docker/cache"
# (docker) : export HF_HOME=/path/to/docker/cache"

docker run \
	--mount "type=bind,src=.,dst=/home/fapannen" \
	--runtime=nvidia \
	--gpus all \
	-it \
	image-generation
