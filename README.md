# noise_cifar10

## Setup
### Clone the repository
```
git clone 
```
### Create the environment
```
docker built -t NAME/IMAGES:TAGS .
```

## How to use
Modify the docker.sh
```
IMAGES=NAME/IMAGES
TAGS=TAGS
```
Launch Docker
```
. docker.sh
```
Change directry
```
cd /path/to/noise_cifar10
```
run program
```
python cifar10_CNN_main.py
```