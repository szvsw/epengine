/// <reference path="./.sst/platform/config.d.ts" />

export default $config({
  app(input) {
    return {
      name: "scytheworkers",
      removal: input?.stage === "production" ? "retain" : "remove",
      protect: ["production"].includes(input?.stage),
      home: "aws",
    };
  },
  async run() {
    const vpc = new sst.aws.Vpc("Vpc");

    const cluster = new sst.aws.Cluster("Cluster", {
      vpc,
    });

    const hatchetToken = new sst.Secret(
      "HATCHET_CLIENT_TOKEN",
      process.env.HATCHET_CLIENT_TOKEN,
    );

    const normalizeName = (name: string, sep: string = "-") => {
      return `${sep === "/" ? "/" : ""}${$app.name}${sep}${
        $app.stage
      }${sep}${name}`;
    };

    const hatchetTokenSecret = new aws.ssm.Parameter(
      normalizeName("HATCHET_CLIENT_TOKEN", "/"),
      {
        type: "SecureString",
        value: hatchetToken.value,
      },
    );

    sst.Linkable.wrap(aws.s3.BucketV2, (bucket) => ({
      properties: { name: bucket.bucket },
      include: [
        sst.aws.permission({
          actions: ["s3:*"],
          resources: [bucket.arn, $interpolate`${bucket.arn}/*`],
        }),
      ],
    }));

    const bucket = process.env.EXISTING_BUCKET
      ? aws.s3.BucketV2.get("Storage", process.env.EXISTING_BUCKET)
      : new sst.aws.Bucket("Storage");

    const bucketName =
      bucket instanceof aws.s3.BucketV2 ? bucket.bucket : bucket.name;

    const args = {
      EP_VERSION: "22.2.0",
      PYTHON_VERSION: "3.12",
      POETRY_VERSION: "2.1.2",
    };
    const simCount = 0;
    const fanCount = 0;
    const trainCount = 0;
    const simService = new sst.aws.Service("Simulations", {
      cluster,
      loadBalancer: undefined,

      cpu: "2 vCPU",
      memory: "8 GB",
      capacity: "spot",
      scaling: {
        min: simCount,
        max: simCount,
      },
      image: {
        dockerfile: "epengine/worker/Dockerfile",
        context: "..",
        args,
      },
      environment: {
        DOES_LEAF: "True",
        DOES_TRAIN: "False",
        DOES_FAN: "False",
        DOES_INFERENCE: "False",
        MAX_RUNS: "1",
      },
      ssm: {
        HATCHET_CLIENT_TOKEN: hatchetTokenSecret.arn,
      },
      link: [bucket],
    });

    const fanoutService = new sst.aws.Service("Fanouts", {
      cluster,
      loadBalancer: undefined,

      cpu: "4 vCPU",
      memory: "16 GB",
      capacity: "spot",
      scaling: {
        min: fanCount,
        max: fanCount,
      },
      image: {
        dockerfile: "epengine/worker/Dockerfile",
        context: "..",
        args,
      },
      environment: {
        DOES_LEAF: "False",
        DOES_TRAIN: "False",
        DOES_FAN: "True",
        DOES_INFERENCE: "False",
        MAX_RUNS: "2",
      },
      ssm: {
        HATCHET_CLIENT_TOKEN: hatchetTokenSecret.arn,
      },
      link: [bucket],
    });

    const trainService = new sst.aws.Service("Train", {
      cluster,
      loadBalancer: undefined,

      cpu: "4 vCPU",
      memory: "16 GB",
      capacity: "spot",
      scaling: {
        min: trainCount,
        max: trainCount,
      },
      image: {
        dockerfile: "epengine/worker/Dockerfile",
        context: "..",
        args,
      },
      environment: {
        DOES_LEAF: "False",
        DOES_TRAIN: "True",
        DOES_FAN: "False",
        DOES_INFERENCE: "False",
        MAX_RUNS: "1",
      },
      ssm: {
        HATCHET_CLIENT_TOKEN: hatchetTokenSecret.arn,
      },
      link: [bucket],
    });
    return {
      bucket: bucketName,
    };
  },
});
