# Deploying the Full Stack

## First Time Setup

Copilot is an official AWS tool used to quickly stand-up infrastructure in the cloud. For this project, it is responsible for setting up a backend service for running simulations (so no inbound internet-access, only outbound) an S3 bucket, and the distributed queuing system. In the future it may also include an API for generating jobs etc, but for now, it will be expected that such work is hosted locally. Note that it is also possible (and easier) to run with a managed version of the distributed queuing system hosted by [Hatchet](cloud.onhatchet.run), which substantially simplifies the setup. Separate docs for that can be found below.

<!-- Once the prereqs are installed, you can either run a single command macro to stand up the whole stack at once, or you can stand-up pieces incrementally. -->

### Prereqs

1. Install [Docker](https://docs.docker.com/get-started/get-docker/).
1. Install [AWS CLI](https://aws.amazon.com/cli/) & [AWS Copilot](https://aws.github.io/copilot-cli/docs/getting-started/install/).
   <!-- 1. Create a [Hatchet account](https://cloud.onhatchet.run/auth/register). -->
   <!-- 1. Generate a [Hatchet API token](https://cloud.onhatchet.run/tenant-settings/api-tokens). _nb: in the future, Hatchet will be deployable as part of the Copilot specification._ -->
1. Configure an AWS CLI profile if you have not done so already: `aws configure --profile <your-profile-name>`
1. You will need a domain in order to access your hosted queue which you can configure CNAME records for. Some Route53 specific instructions may be added in the future, but for now it is assumed that you will manually configure the CNAME records at the indicated point in the process.
1. Clone the repository: `git clone https://github.com/szvsw/epengine`.
1. Move into the repository: `cd epengine`

### Standing up the environment

1. `copilot app init`
   1. see [`app init` help](https://aws.github.io/copilot-cli/docs/commands/app-init/) for more details like `--permissions-boundary` if needed
1. `copilot env init --name prod`
   1. (see [`see init` help](https://aws.github.io/copilot-cli/docs/commands/env-init/) for more details, as well as [`init` manifest help](https://aws.github.io/copilot-cli/docs/manifest/environment/) for specs for VPCs etc.)
1. _TODO: instructions about `CERTIFICATE_ARN` env var interpolation._
1. The next command will deploy an Aurora Serverless PostgresSql database, S3 bucket, and RabbitMQ broker.
   1. You may wish to turn on RabbitMQ console public accessibilty (_ADD HOW_)
   1. You may wish to set the instance sizes for RabbitMQ.
   1. You may wish to update the scaling capacities for Aurora Serverless.
1. `copilot env deploy --name prod`
1. `copilot svc init --name hatchet-engine`
1. _TODO: instructions about `API_DOMAIN` env var interpolation._
1. `copilot svc deploy --name hatchet-engine --env prod`
1. _TODO: instructions about the `FS_ID` and `FSAP_ID` env var interpolation._
1. `copilot svc init --name hatchet-api`
1. `copilot svc deploy --name hatchet-api --env prod`
1. Create a `CNAME` record to point at the load balancer (_TODO: how to find the load balancer address_).
1. Visit your domain and either login with a new account & make a tenant, or login with the default creds: `admin@example.com` / `Admin123!!`
1. Make an api token (General Settings > API Tokens). _TODO: (automate token creation steps with a task run)_
1. `copilot secret init --name HATCHET_CLIENT_TOKEN --overwrite` then paste in your secret.
<!-- 1. disable/enable TLS -->
1. `copilot svc init --name hatchet-worker`
1. Authenticate to AWS ECR (_TODO: add instructions_).
<!-- 1. `make docker-login` -->
1. `copilot svc deploy --name hatchet-worker --env prod`

## Scaling workers up and down

In `copilot/hatchet-worker/manifest.yml`, you can edit the worker count configuration to determine how many workers you would like, if scaling should be done based off of CPU utilization, etc.

After making changes, run `copilot svc deploy --name hatchet-worker --env prod`.

## Cleanup

1. Empty your buckets if they have files in them.
1. `copilot app delete`
