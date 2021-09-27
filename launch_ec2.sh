

aws ec2 run-instances \
    --image-id ami-02cf3af4882d6824b \
    --security-group-ids sg-0da7694c867e84ae3 \
    --count 1 \
    --instance-type r5.2xlarge \
    --key-name thelio

aws ec2 describe-instances \
  --query "Reservations[*].Instances[*].PublicIpAddress" \
  --output=text

aws ec2 describe-instances \
  --query "Reservations[*].Instances[*].PublicDnsName" \
  --output=text

aws ec2 describe-images --owner self

