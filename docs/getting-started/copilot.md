# Copilot Deployment

## First Time Setup

### Prereqs

1. Install [Docker](https://docs.docker.com/get-started/get-docker/).
1. Install [AWS CLI](https://aws.amazon.com/cli/) & [AWS Copilot](https://aws.github.io/copilot-cli/docs/getting-started/install/).
1. Create a [Hatchet account](https://cloud.onhatchet.run/auth/register).
1. Generate a [Hatchet API token](https://cloud.onhatchet.run/tenant-settings/api-tokens). _nb: in the future, Hatchet will be deployable as part of the Copilot specification._
1. Configure an AWS CLI profile if you have not done so already: `aws configure --profile <your-profile-name>`

Once the prereqs are installed, you can either run a single command macro to stand up the whole stack at once, or you can stand-up pieces incrementally.

### Command macro (all at once)

Running the following command will walk you through all the steps necessary to stand-up the application in AWS.

```bash
make create-copilot-app
```

You will be prompted to select a few things:

1. A name for the application.
1. The AWS profile to use.
1. Any configuration necessary for the environment (e.g. security groups, VPCs, subnets etc), however you can use the defaults if desired (recommended for most users).
1. The Hatchet Client Token

### Manually running the commands

1. `copilot app init`
   1. see [`app init` help](https://aws.github.io/copilot-cli/docs/commands/app-init/) for more details like `--permissions-boundary` if needed
1. `copilot env init --name prod`
   1. (see [`see init` help](https://aws.github.io/copilot-cli/docs/commands/env-init/) for more details, as well as [`init` manifest help](https://aws.github.io/copilot-cli/docs/manifest/environment/) for specs for VPCs etc.)
1. `copilot env deploy --name prod`
1. `copilot secret init --name HATCHET_CLIENT_TOKEN`
1. `copilot deploy --init-wkld --env prod --all`

## Managing Worker Counts

In `copilot/worker/manifest.yml`, you can edit the worker count configuration to determine how many workers you would like, if scaling should be done based off of CPU utilization, etc.

After making changes, run `copilot svc deploy`.

## Cleanup

1. Empty your buckets if they have files in them.
1. `copilot app delete`
