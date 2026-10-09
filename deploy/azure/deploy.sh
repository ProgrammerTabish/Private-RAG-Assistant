#!/usr/bin/env bash
# Deploy the privrag Docker stack to Azure - run in Azure Cloud Shell (bash) from the repo root:
#
#   bash deploy/azure/deploy.sh
#
# What it does (safe to re-run; every step skips what already exists):
#   1. LLM:   Llama-3.3-70B-Instruct deployment in the existing AI Foundry resource (pay per token)
#   2. Image: builds the backend image in Azure Container Registry (no local Docker needed)
#   3. VM:    Ubuntu CPU VM with Docker in its own VNet (existing VNets/resources are not touched)
#   4. Stack: writes docker-compose.yml + .env on the VM and starts it
#             (qdrant + BGE-M3 embeddings + one-time indexer + api on port 8000)
#   5. Access: port 8000 only from the existing web app's outbound IPs (+ ALLOW_IPS)
#
# Override any setting with environment variables, e.g. VM_SIZE=Standard_D4_v2 bash deploy/azure/deploy.sh
set -euo pipefail

RG=${RG:-rg-challenge-rag}
LOC=${LOC:-swedencentral}
FOUNDRY=${FOUNDRY:-spgdehakaton-6670-resource}
LLM_MODEL=${LLM_MODEL:-Llama-3.3-70B-Instruct}
LLM_DEPLOYMENT=${LLM_DEPLOYMENT:-$LLM_MODEL}
LLM_VERSION=${LLM_VERSION:-9}
LLM_CAPACITY=${LLM_CAPACITY:-50}                 # thousands of tokens per minute
ACR=${ACR:-rgcontainerrag}
IMAGE_TAG=${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M)}
VM=${VM:-vm-privrag}
VM_SIZES=${VM_SIZES:-"Standard_D8ds_v6 Standard_D8ds_v7 Standard_D4_v2"}   # tried in this order
WEBAPP=${WEBAPP:-challenge-rag-app}             # its outbound IPs may call the API
ALLOW_IPS=${ALLOW_IPS:-}                        # extra IPs/CIDRs, comma separated (e.g. your office IP)
SKIP_START=${SKIP_START:-false}                 # true = prepare everything but do not start the stack

step() { echo; echo "==> $*"; }
[ -f docker-compose.yml ] || { echo "run this from the repo root (docker-compose.yml not found)"; exit 1; }

# ---------------------------------------------------------------- 1 LLM in AI Foundry
step "1/5 LLM deployment '$LLM_DEPLOYMENT' in $FOUNDRY"
if az cognitiveservices account deployment show -n "$FOUNDRY" -g "$RG" --deployment-name "$LLM_DEPLOYMENT" -o none 2>/dev/null; then
  echo "exists"
else
  az cognitiveservices account deployment create -n "$FOUNDRY" -g "$RG" --deployment-name "$LLM_DEPLOYMENT" \
    --model-name "$LLM_MODEL" --model-version "$LLM_VERSION" --model-format Meta \
    --sku-name GlobalStandard --sku-capacity "$LLM_CAPACITY" -o none
  echo "created"
fi
LLM_KEY=$(az cognitiveservices account keys list -n "$FOUNDRY" -g "$RG" --query key1 -o tsv)
LLM_BASE_URL="https://${FOUNDRY}.openai.azure.com/openai/v1"
echo "endpoint: $LLM_BASE_URL  (model: $LLM_DEPLOYMENT)"
for i in 1 2 3 4 5 6; do
  code=$(curl -s -o /tmp/llm_test.json -w '%{http_code}' "$LLM_BASE_URL/chat/completions" \
    -H "Content-Type: application/json" -H "api-key: $LLM_KEY" -H "Authorization: Bearer $LLM_KEY" \
    -d "{\"model\":\"$LLM_DEPLOYMENT\",\"messages\":[{\"role\":\"user\",\"content\":\"Antworte nur mit OK\"}],\"max_tokens\":5}")
  [ "$code" = "200" ] && break
  echo "  LLM test HTTP $code - retrying in 20 s (new deployments need a moment)"; sleep 20
done
[ "$code" = "200" ] || { echo "LLM test failed:"; cat /tmp/llm_test.json; exit 1; }
echo "LLM answers: $(python3 -c "import json;print(json.load(open('/tmp/llm_test.json'))['choices'][0]['message']['content'])")"

# ---------------------------------------------------------------- 2 image in ACR
step "2/5 Build backend image $ACR.azurecr.io/privrag-api:$IMAGE_TAG"
if az acr repository show-tags -n "$ACR" --repository privrag-api -o tsv 2>/dev/null | grep -qx "$IMAGE_TAG"; then
  echo "image tag exists"
else
  az acr build -r "$ACR" -t "privrag-api:$IMAGE_TAG" -t privrag-api:latest . -o none
fi
ACR_SERVER=$(az acr show -n "$ACR" --query loginServer -o tsv)
ACR_USER=$(az acr credential show -n "$ACR" --query username -o tsv)
ACR_PASS=$(az acr credential show -n "$ACR" --query 'passwords[0].value' -o tsv)

# ---------------------------------------------------------------- 3 VM
step "3/5 VM $VM"
if az vm show -g "$RG" -n "$VM" -o none 2>/dev/null; then
  echo "exists"
else
  created=false
  for size in $VM_SIZES; do
    echo "trying $size ..."
    ctrl=(); case "$size" in *_v6|*_v7) ctrl=(--disk-controller-type NVMe);; esac
    if az vm create -g "$RG" -n "$VM" -l "$LOC" --size "$size" \
        --image Canonical:ubuntu-24_04-lts:server:latest "${ctrl[@]}" \
        --admin-username azureuser --generate-ssh-keys \
        --os-disk-size-gb 128 --storage-sku Premium_LRS \
        --public-ip-sku Standard --nsg-rule NONE -o none; then
      created=true; echo "created with $size"; break
    fi
  done
  $created || { echo "could not create a VM with any of: $VM_SIZES"; exit 1; }
fi
VM_IP=$(az vm show -d -g "$RG" -n "$VM" --query publicIps -o tsv)

# ---------------------------------------------------------------- 4 stack on the VM
step "4/5 Docker stack on the VM (via run-command, no SSH needed)"
API_KEY_FILE=~/.privrag_api_key
[ -s "$API_KEY_FILE" ] || openssl rand -hex 24 > "$API_KEY_FILE"
PRIVRAG_API_KEY=$(cat "$API_KEY_FILE")
ENV_CONTENT="LLM_BASE_URL=$LLM_BASE_URL
LLM_MODEL=$LLM_DEPLOYMENT
LLM_API_KEY=$LLM_KEY
API_IMAGE=$ACR_SERVER/privrag-api:$IMAGE_TAG
PRIVRAG_API_KEY=$PRIVRAG_API_KEY
PRIVRAG_TOP_K=8
PRIVRAG_QUERY_REWRITE=true
PRIVRAG_LLM_TIMEOUT_S=120
PRIVRAG_LLM_MAX_TOKENS=900
PRIVRAG_EMBED_BATCH_SIZE=16
PRIVRAG_EMBED_TIMEOUT_S=300"
COMPOSE_B64=$(base64 -w0 docker-compose.yml)
ENV_B64=$(printf '%s\n' "$ENV_CONTENT" | base64 -w0)
# compose waits for the indexer (hours on CPU) before starting the api, even with -d;
# run it in the background so run-command (90 min limit) returns immediately
START_CMD='nohup docker compose up -d --no-build > /opt/privrag/up.log 2>&1 & sleep 20; tail -5 /opt/privrag/up.log'
[ "$SKIP_START" = "true" ] && START_CMD='echo "SKIP_START=true - stack prepared, not started"'
VM_SCRIPT=$(cat <<EOS
set -e
if ! command -v docker >/dev/null; then curl -fsSL https://get.docker.com | sh; fi
usermod -aG docker azureuser || true
mkdir -p /opt/privrag && cd /opt/privrag
echo '$COMPOSE_B64' | base64 -d > docker-compose.yml
echo '$ENV_B64' | base64 -d > .env && chmod 600 .env
echo '$ACR_PASS' | docker login $ACR_SERVER -u $ACR_USER --password-stdin
docker compose pull
$START_CMD
docker compose ps
EOS
)
az vm run-command invoke -g "$RG" -n "$VM" --command-id RunShellScript --scripts "$VM_SCRIPT" \
  --query 'value[0].message' -o tsv | tail -25

# ---------------------------------------------------------------- 5 network access to port 8000
step "5/5 Allow port 8000 from $WEBAPP outbound IPs ${ALLOW_IPS:+and $ALLOW_IPS}"
NSG=$(az network nic show --ids "$(az vm show -g "$RG" -n "$VM" --query 'networkProfile.networkInterfaces[0].id' -o tsv)" \
      --query 'networkSecurityGroup.id' -o tsv | awk -F/ '{print $NF}')
SOURCES=$(az webapp show -n "$WEBAPP" -g "$RG" --query possibleOutboundIpAddresses -o tsv 2>/dev/null || true)
[ -n "$ALLOW_IPS" ] && SOURCES="${SOURCES:+$SOURCES,}$ALLOW_IPS"
if [ -n "$SOURCES" ]; then
  az network nsg rule create -g "$RG" --nsg-name "$NSG" -n allow-api-8000 --priority 1000 \
    --direction Inbound --access Allow --protocol Tcp --destination-port-ranges 8000 \
    --source-address-prefixes $(echo "$SOURCES" | tr ',' ' ') -o none
  echo "NSG $NSG: port 8000 open for $(echo "$SOURCES" | tr ',' '\n' | wc -l) source address(es)"
else
  echo "no source addresses - port 8000 stays closed (set ALLOW_IPS)"
fi

echo
echo "DONE."
echo "  API:      http://$VM_IP:8000   (header X-API-Key: see  cat $API_KEY_FILE  - keep it secret)"
echo "  Indexing: az vm run-command invoke -g $RG -n $VM --command-id RunShellScript \\"
echo "              --scripts 'cd /opt/privrag && docker compose logs --tail 5 indexer' --query 'value[0].message' -o tsv"
echo "  Health:   az vm run-command invoke -g $RG -n $VM --command-id RunShellScript \\"
echo "              --scripts 'curl -s localhost:8000/health' --query 'value[0].message' -o tsv"
echo "  Stop VM (saves cost, data kept):  az vm deallocate -g $RG -n $VM"
