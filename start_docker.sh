#!/bin/bash
CONTAINER_NAME="digit_rec"
grep_container=$(docker images | grep -i "${CONTAINER_NAME}")
len_str=${#grep_container}

check_container(){
	if [ $len_str -ne 0 ]
		then
			return 0
		else
			return 1
	fi
}

build_and_run(){
	docker build -t "${CONTAINER_NAME}" . &&
	docker run -itd --name "${CONTAINER_NAME}" -v $(pwd):/app "${CONTAINER_NAME}"
}


if check_container
	then docker start "${CONTAINER_NAME}"
	else build_and_run 
fi

