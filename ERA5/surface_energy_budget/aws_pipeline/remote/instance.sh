#!/usr/bin/env bash
# Laptop: status / stop / start / terminate the instance from launch_instance.sh.
#   ./instance.sh status | stop | start | terminate | ssh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REGION=us-west-2
[ -f "$HERE/.last_instance" ] || { echo "no .last_instance here: run launch_instance.sh first"; exit 1; }
read -r INSTANCE_ID DNS KEY_FILE < "$HERE/.last_instance"
case "${1:-status}" in
  status)    aws ec2 describe-instances --region $REGION --instance-ids "$INSTANCE_ID" \
               --query 'Reservations[0].Instances[0].[InstanceId,InstanceType,State.Name,PublicDnsName]' --output text ;;
  stop)      aws ec2 stop-instances --region $REGION --instance-ids "$INSTANCE_ID" --output text ;;
  start)     aws ec2 start-instances --region $REGION --instance-ids "$INSTANCE_ID" --output text
             aws ec2 wait instance-running --region $REGION --instance-ids "$INSTANCE_ID"
             DNS="$(aws ec2 describe-instances --region $REGION --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].PublicDnsName' --output text)"
             echo "$INSTANCE_ID $DNS $KEY_FILE" > "$HERE/.last_instance"; echo "ssh -i $KEY_FILE ubuntu@$DNS" ;;
  terminate) read -r -p "terminate $INSTANCE_ID and delete its disk? [y/N] " a; [ "$a" = y ] || exit 0
             # A persistent spot request would relaunch a replacement: cancel it first.
             SIR="$(aws ec2 describe-instances --region $REGION --instance-ids "$INSTANCE_ID" \
                     --query 'Reservations[0].Instances[0].SpotInstanceRequestId' --output text)"
             if [ -n "$SIR" ] && [ "$SIR" != None ]; then
               aws ec2 cancel-spot-instance-requests --region $REGION --spot-instance-request-ids "$SIR" --output text
             fi
             aws ec2 terminate-instances --region $REGION --instance-ids "$INSTANCE_ID" --output text ;;
  ssh)       exec ssh -i "$KEY_FILE" "ubuntu@$DNS" ;;
  *)         echo "usage: $0 status|stop|start|terminate|ssh"; exit 1 ;;
esac
