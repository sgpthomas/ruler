# some other commands
rsync -rhP -e "ssh -i ~/.ssh/thelio.pem" ~/Research/ruler/ ubuntu@18.222.147.140:~/ruler/ --exclude={.git,target}

rsync -rhP -e "ssh -i ~/.ssh/thelio.pem" ~/Research/dios/ ubuntu@18.222.147.140:~/diospyros/ --exclude={.git,target}
