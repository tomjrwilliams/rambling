## download packages

deb https://nginx.org/packages/ubuntu/ focal nginx
deb-src https://nginx.org/packages/ubuntu/ focal nginx

## To install the packages, execute in your shell:

sudo apt update
sudo apt install nginx

## If a W: GPG error: https://nginx.org/packages/ubuntu focal InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY $key is encountered during the NGINX repository update, execute the following:

## Replace $key with the corresponding $key from your GPG error.

sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys $key
sudo apt update
sudo apt install nginx

## You have now nginx installed on your server but not ready to serve web pages. you have to start the nginx. You can do this by using this command:

sudo systemctl start nginx




# NOTE:
# replace the python server on linux with just install nginx and static web server?