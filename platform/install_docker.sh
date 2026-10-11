#!/usr/bin/env bash
# Install Docker Engine (Apache-2.0, from Docker's official apt repo) inside WSL Ubuntu.
# Run as root: wsl -d Ubuntu-24.04 -u root -- bash platform/install_docker.sh <user>
set -euo pipefail
USER_NAME="${1:?usage: install_docker.sh <linux-user>}"

apt-get update -qq
apt-get install -y -qq ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
. /etc/os-release
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
usermod -aG docker "$USER_NAME"
systemctl enable --now docker
docker version --format 'docker {{.Server.Version}}'
docker compose version
