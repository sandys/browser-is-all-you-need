#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

PROJECT_ID="${GLM47_GCP_PROJECT_ID:-transformer-tales}"
REGION="${GLM47_GCP_REGION:-us-central1}"
RUNNER_PRINCIPAL="${GLM47_GCP_RUNNER_PRINCIPAL:-}"
ASSET_WRITER_PRINCIPAL="${GLM47_GCP_ASSET_WRITER_PRINCIPAL:-}"
ROLE_ID="${GLM47_GCP_RUNNER_ROLE_ID:-glm47SkyPilotRunner}"
SERVICE_ACCOUNT_ID="${GLM47_GCP_SERVICE_ACCOUNT_ID:-skypilot-v1}"
MODELS_BUCKET="${GLM47_GCP_MODELS_BUCKET:-${PROJECT_ID}-glm47-models}"
ASSETS_BUCKET="${GLM47_GCP_ASSETS_BUCKET:-${PROJECT_ID}-glm47-assets}"
RUNS_BUCKET="${GLM47_GCP_RUNS_BUCKET:-${PROJECT_ID}-glm47-runs}"
ROLE_FILE="${SCRIPT_DIR}/iam/skypilot-runner-role.yaml"

if [ -z "${RUNNER_PRINCIPAL}" ]; then
  echo "Set GLM47_GCP_RUNNER_PRINCIPAL to user:, group:, domain:, or serviceAccount:." >&2
  exit 2
fi

case "${RUNNER_PRINCIPAL}" in
  user:* | group:* | domain:* | serviceAccount:*) ;;
  *)
    echo "Unsupported runner principal: ${RUNNER_PRINCIPAL}" >&2
    exit 2
    ;;
esac

if [ -n "${ASSET_WRITER_PRINCIPAL}" ]; then
  case "${ASSET_WRITER_PRINCIPAL}" in
    user:* | group:* | domain:* | serviceAccount:*) ;;
    *)
      echo "Unsupported asset-writer principal: ${ASSET_WRITER_PRINCIPAL}" >&2
      exit 2
      ;;
  esac
fi

for command in gcloud; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Missing required command: ${command}" >&2
    exit 2
  fi
done

SERVICE_ACCOUNT="${SERVICE_ACCOUNT_ID}@${PROJECT_ID}.iam.gserviceaccount.com"
CUSTOM_ROLE="projects/${PROJECT_ID}/roles/${ROLE_ID}"

gcloud services enable \
  cloudresourcemanager.googleapis.com \
  compute.googleapis.com \
  iam.googleapis.com \
  serviceusage.googleapis.com \
  storage.googleapis.com \
  --project="${PROJECT_ID}"

if gcloud iam roles describe "${ROLE_ID}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud iam roles update "${ROLE_ID}" \
    --file="${ROLE_FILE}" \
    --project="${PROJECT_ID}"
else
  gcloud iam roles create "${ROLE_ID}" \
    --file="${ROLE_FILE}" \
    --project="${PROJECT_ID}"
fi

if ! gcloud iam service-accounts describe "${SERVICE_ACCOUNT}" \
  --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${SERVICE_ACCOUNT_ID}" \
    --display-name="Shared SkyPilot runtime" \
    --project="${PROJECT_ID}"
fi

for MEMBER in "${RUNNER_PRINCIPAL}" "serviceAccount:${SERVICE_ACCOUNT}"; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="${MEMBER}" \
    --role="${CUSTOM_ROLE}" \
    --condition=None
done

gcloud iam service-accounts add-iam-policy-binding "${SERVICE_ACCOUNT}" \
  --member="${RUNNER_PRINCIPAL}" \
  --role=roles/iam.serviceAccountUser \
  --condition=None \
  --project="${PROJECT_ID}"

for BUCKET in "${MODELS_BUCKET}" "${ASSETS_BUCKET}" "${RUNS_BUCKET}"; do
  if ! gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
    gcloud storage buckets create "gs://${BUCKET}" \
      --location="${REGION}" \
      --project="${PROJECT_ID}" \
      --uniform-bucket-level-access
  fi
done

for MEMBER in "${RUNNER_PRINCIPAL}" "serviceAccount:${SERVICE_ACCOUNT}"; do
  for BUCKET in "${MODELS_BUCKET}" "${ASSETS_BUCKET}"; do
    gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
      --member="${MEMBER}" \
      --role=roles/storage.objectViewer \
      --condition=None
  done
  gcloud storage buckets add-iam-policy-binding "gs://${RUNS_BUCKET}" \
    --member="${MEMBER}" \
    --role=roles/storage.objectUser \
    --condition=None
done

if [ -n "${ASSET_WRITER_PRINCIPAL}" ]; then
  for BUCKET in "${MODELS_BUCKET}" "${ASSETS_BUCKET}"; do
    gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
      --member="${ASSET_WRITER_PRINCIPAL}" \
      --role=roles/storage.objectUser \
      --condition=None
  done
fi

echo "Shared GCP resources are configured."
echo "project=${PROJECT_ID}"
echo "region=${REGION}"
echo "runtime_service_account=${SERVICE_ACCOUNT}"
echo "models_bucket=gs://${MODELS_BUCKET}"
echo "assets_bucket=gs://${ASSETS_BUCKET}"
echo "runs_bucket=gs://${RUNS_BUCKET}"
