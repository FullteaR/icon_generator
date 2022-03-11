# icon_generator
This is an automated animator for drawing anime/manga face using stylegan3.

## Generated Images

![sample1](https://github.com/FullteaOfEEIC/icon_generator/blob/master/sample_images/sample1.png)
![sample2](https://github.com/FullteaOfEEIC/icon_generator/blob/master/sample_images/sample2.png)
![sample3](https://github.com/FullteaOfEEIC/icon_generator/blob/master/sample_images/sample3.png)
![sample4](https://github.com/FullteaOfEEIC/icon_generator/blob/master/sample_images/sample4.png)

## Installation and Usage

### Requirements

- docker
- docker-compose

### Steps

#### Clone this repository
```
git clone https://github.com/FullteaOfEEIC/icon_generator.git
```

#### Save Images
Save image files to ```input```. (Datasets and scripts for collecting datasets are not included.)

#### Use docker-compose
```
docker-compose up -d
```
This will automatically detect faces from images and save infos to ```output/output.json```.

#### Access jupyter

Open your web browser and access http://localhost:11112/notebooks.
With running trim.ipynb, trimed images are saved to ```faces```.

(you can specify another port by editing ```docker-compose.yml```)

#### Run stylegan3

##### convert image 
Convert images from image file to tf-records.
The following script creates tf-records to ```datasets``` folder from images saved in ```faces```
```
docker exec -it style python dataset_tool.py --source=/faces --dest=/datasets/faces.zip
```
##### training
```
docker exec -it style python train.py --outdir=/results --cfg=stylegan3-t --data=/datasets/faces.zip --gpus=2 --batch=16 --gamma=8.2 --mirror=1
```
for more details about options, please see https://github.com/NVlabs/stylegan3
