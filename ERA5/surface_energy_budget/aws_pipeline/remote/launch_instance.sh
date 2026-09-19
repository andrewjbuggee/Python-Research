#!/usr/bin/env bash
# Launch an EC2 instance in us-west-2 (the bucket's region) ready to run the analysis.
#
# One-time prerequisites on the laptop (see README "Setting up AWS"):
#   - an AWS account, and `aws configure` done with an access key that may
#     create EC2 instances, key pairs and security groups
#   - `brew install awscli` (or pip install awscli)
#
# Usage:
#   ./launch_instance.sh                         # c7i.4xlarge, 300 GB, on-demand
#   INSTANCE_TYPE=r7i.4xlarge SPOT=1 ./launch_instance.sh
#
# Prints the instance id and the ssh command. The bootstrap (bootstrap.sh) runs
# automatically at first boot via user-data; give it ~5 minutes, then
#   ssh -i ~/.ssh/era5-seb.pem ubuntu@<public-dns>  'tail -f /var/log/cloud-init-output.log'
#
# Sizing notes (us-west-2 on-demand prices are approximate, check the console):
#   c7i.4xlarge   16 vCPU  32 GB   ~$0.71/h   default: decoding is CPU-bound in-region
#   r7i.4xlarge   16 vCPU 128 GB   ~$1.06/h   for the arctic_circle box (136,800 cells)
#   c7i.8xlarge   32 vCPU  64 GB   ~$1.43/h   halves the wall time of a big pull
#   SPOT=1        the same at ~30-40% of the price; may be reclaimed (the chunk
#                 cache on the root volume survives a stop but not a termination).
set -euo pipefail

REGION="us-west-2"
INSTANCE_TYPE="${INSTANCE_TYPE:-c7i.4xlarge}"
VOLUME_GB="${VOLUME_GB:-300}"
KEY_NAME="${KEY_NAME:-era5-seb}"
KEY_FILE="${KEY_FILE:-$HOME/.ssh/${KEY_NAME}.pem}"
SG_NAME="${SG_NAME:-era5-seb-ssh}"
NAME_TAG="${NAME_TAG:-era5-seb}"
SPOT="${SPOT:-0}"
REPO_URL="${REPO_URL:-https://github.com/andrewjbuggee/Python-Research.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
HERE="$(cd "$(dirname "$0")" && pwd)"

command -v aws >/dev/null || { echo "aws CLI not found: brew install awscli && aws configure"; exit 1; }
aws sts get-caller-identity --region "$REGION" >/dev/null || { echo "aws credentials not configured (aws configure)"; exit 1; }

# --- preflight: the instance clones GitHub main; anything not pushed (this
# directory itself, today's edits) and anything gitignored (the ARM
# observation .xlsx) reach it only through sync_code.sh. Warn, do not stop.
SEB_LOCAL="$(cd "$HERE/../.." && pwd)"
if ! git -C "$SEB_LOCAL" ls-files --error-unmatch aws_pipeline/era5_s3.py >/dev/null 2>&1 \
   || [ -n "$(git -C "$SEB_LOCAL" status --porcelain -- . 2>/dev/null)" ]; then
  echo "note: uncommitted or unpushed changes under $SEB_LOCAL -- after the instance boots, run"
  echo "      ./sync_code.sh   to rsync the working tree (incl. gitignored inputs) onto it."
fi

# --- key pair -------------------------------------------------------------
if ! aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" >/dev/null 2>&1; then
  echo "creating key pair $KEY_NAME -> $KEY_FILE"
  aws ec2 create-key-pair --region "$REGION" --key-name "$KEY_NAME" --key-type ed25519 \
    --query KeyMaterial --output text > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
fi
[ -f "$KEY_FILE" ] || { echo "key pair exists in AWS but $KEY_FILE is missing; set KEY_NAME to a new name"; exit 1; }

# --- security group: SSH from this laptop's public IP only ------------------
MY_IP="$(curl -s https://checkip.amazonaws.com)/32"
VPC_ID="$(aws ec2 describe-vpcs --region "$REGION" --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)"
SG_ID="$(aws ec2 describe-security-groups --region "$REGION" --filters Name=group-name,Values="$SG_NAME" Name=vpc-id,Values="$VPC_ID" \
          --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)"
if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
  SG_ID="$(aws ec2 create-security-group --region "$REGION" --group-name "$SG_NAME" \
            --description "ssh for era5-seb" --vpc-id "$VPC_ID" --query GroupId --output text)"
fi
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG_ID" \
  --protocol tcp --port 22 --cidr "$MY_IP" >/dev/null 2>&1 || true   # already present is fine

# --- latest Ubuntu 24.04 LTS AMI (Canonical's public SSM parameter) -----------
AMI="$(aws ssm get-parameter --region "$REGION" \
        --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
        --query Parameter.Value --output text)"

# --- user-data: run bootstrap.sh as ubuntu at first boot ------------------------
USER_DATA="$(mktemp)"
{
  echo '#!/bin/bash'
  echo 'set -eux'
  echo "cat > /home/ubuntu/bootstrap.sh <<'BOOT'"
  cat "$HERE/bootstrap.sh"
  echo 'BOOT'
  echo "cat > /home/ubuntu/environment.yml <<'ENVYML'"
  cat "$HERE/environment.yml"
  echo 'ENVYML'
  echo 'chown ubuntu:ubuntu /home/ubuntu/bootstrap.sh /home/ubuntu/environment.yml'
  echo "sudo -u ubuntu -H env REPO_URL='$REPO_URL' REPO_BRANCH='$REPO_BRANCH' bash /home/ubuntu/bootstrap.sh > /home/ubuntu/bootstrap.log 2>&1"
} > "$USER_DATA"

# A PERSISTENT spot request so the instance can be stopped and started with
# its disk (and chunk cache) intact. Consequence: terminating the instance
# alone would let EC2 launch a replacement -- instance.sh terminate cancels
# the request first.
MARKET=()
if [ "$SPOT" = "1" ]; then
  MARKET=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=persistent,InstanceInterruptionBehavior=stop}')
fi

echo "launching $INSTANCE_TYPE ($AMI) in $REGION, ${VOLUME_GB} GB gp3, spot=$SPOT"
INSTANCE_ID="$(aws ec2 run-instances --region "$REGION" \
  --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --user-data "file://$USER_DATA" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME_TAG}]" \
  ${MARKET[@]+"${MARKET[@]}"} \
  --query 'Instances[0].InstanceId' --output text)"
rm -f "$USER_DATA"

echo "instance $INSTANCE_ID starting..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"
DNS="$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicDnsName' --output text)"
cat <<MSG

instance : $INSTANCE_ID   ($INSTANCE_TYPE, $REGION)
ssh      : ssh -i $KEY_FILE ubuntu@$DNS
bootstrap: runs automatically; follow it with
           ssh -i $KEY_FILE ubuntu@$DNS 'tail -f bootstrap.log'
sync code: $HERE/sync_code.sh            (rsync this working tree + observation inputs onto it)
stop     : $HERE/instance.sh stop         (keeps the disk + cache)
terminate: $HERE/instance.sh terminate    (cancels the spot request too; deletes everything)

Remember: a running instance bills by the second whether or not it is working.
MSG
echo "$INSTANCE_ID $DNS $KEY_FILE" > "$HERE/.last_instance"
