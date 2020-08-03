# FasterRCNN endpoint

Endpoint to detect objects with Faster-RCNN.
I built this project to use it with [coco-annotator](https://github.com/jsbroks/coco-annotator).

## Warning
This implementation is meant to be use in a local network.
There is no control over who uses this endpoint.
Do not expose it to the internet.

If you want to do so, you will probably want to use flask_login and the login_required decorator.

## Install

If you want to use the GPU got to [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and follow the instructions

Clone this repository and go to the new folder:
```bash
git clone git@github.com:ababino/frcnn_endpoint.git
```

To use the default parameters (Faster R-CNN ResNet101 trained on the COCO dataset) download the model using the script:
```bash
bash ./frcnn_model.sh
```

Build
```bash
docker build -t youruser/frcnn_api .
```

pull the image from docker hub:
```bash

```

## Running

To run the container execute:
```bash
docker run -p 4020:4020 --gpus all --env-file ./env.list -v $(pwd)/models:/models ababino/frcnn_api:latest
```

To test the server you can use any image and curl:

```bash
curl -i -X POST  http://localhost:4020/frcnn/  -F "image=@test_image.png"
```


## Reverse Proxy

The following configuration works on Nginx:

```
server {
    listen 80;
    server_name mysitename;

    access_log  /var/log/nginx/access.log;
    error_log  /var/log/nginx/error.log;

    client_max_body_size 10M;

    location / {
        add_header 'Access-Control-Allow-Origin' '*';
        proxy_pass http://localhost:4020;
    }
}
```
