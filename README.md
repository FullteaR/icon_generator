# icon_generator
This is an automated animator for drawing anime/manga face using stylegan2.

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
