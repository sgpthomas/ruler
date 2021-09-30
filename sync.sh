set -x

ADDR=3.144.120.248

# some other commands
dir=ruler
rsync -rP --exclude=.git \
      --exclude=target \
      -e "ssh -i ~/.ssh/thelio.pem" \
      ~/Research/$dir/ "ubuntu@$ADDR:~/$dir/"
      
dir=diospyros
rsync -rP --exclude=.git \
      --exclude=target \
      -e "ssh -i ~/.ssh/thelio.pem" \
      ~/Research/$dir/ "ubuntu@$ADDR:~/$dir/"
