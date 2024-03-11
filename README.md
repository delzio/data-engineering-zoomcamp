# data-engineering-zoomcamp

### setting up vm in gcp:
 - create new vm in gcp
 - on local machine create ssh keys: "ssh-keygen -t rsa -f ~/.ssh/KEY_FILENAME -C USER -b 2048
 - upload public ssh key to gcp: "compute engine, settings, metadata, ssh keys - copy public key"
 - connect using external ip
 - set up config name:
    - cd .ssh
    - touch config
