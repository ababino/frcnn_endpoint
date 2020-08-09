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
cd frcnn_endpoint
```

To use the default parameters ([Faster R-CNN ResNet101 trained on the COCO dataset](https://github.com/ababino/pytorch_faster_rcnn_resnet101)) download the model using the script:
```bash
bash ./frcnn_model.sh
```
If you want to use any other model (such as Mask-RCNN), you will have to download it and update the environmental variables in the `env.list` file.

Pull the image from docker hub:
```bash
docker pull ababino85/frcnn_endpoint:latest
```

## Running

To run the container execute:
```bash
docker run -p 4020:4020 --gpus all --env-file ./env.list -v $(pwd)/models:/models ababino85/frcnn_endpoint:latest
```

To test the server you can use any image and curl.
Open a new terminal, go to the repository folder and execute:

```bash
curl -i -X POST  http://localhost:4020/frcnn/  -F "image=@test_image.png"
```

You should see this output:
```
HTTP/1.1 200 OK
Server: gunicorn/19.10.0
Date: Mon, 03 Aug 2020 22:48:33 GMT
Connection: close
Content-Type: application/json
Content-Length: 336
Access-Control-Allow-Origin: *

{"coco": {"images": [{"width": 640, "height": 451}], "categories": [{"id": 19, "name": "horse"}, {"id": 1, "name": "person"}], "annotations": [{"segmentation": [[122, 123, 122, 411, 495, 411, 495, 123]], "category_id": 19, "isbbox": true}, {"segmentation": [[237, 53, 237, 280, 332, 280, 332, 53]], "category_id": 1, "isbbox": true}]}}
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
