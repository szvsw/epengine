# sequence

1. copilot env init --name prod
1. consider turning on rabbit mq console public accessibilty, setting instance sizes for rabbit mq
1. copilot env deploy --name prod
1. copilot svc init --name hatchet-engine
1. copilot svc deploy --name hatchet-engine --env prod
1. update the the volume mount fs-id and fsap-id
1. update the alias and import the certificate arn
1. copilot svc init --name hatchet-api
1. copilot svc deploy --name hatchet-api --env prod
1. update your cname record to point at the load balancer
1. log in
1. make a tenant
1. or login with `admin@example.com` / `Admin123!!`
1. make an api token
1. (these steps can potentially be automated with a task run)
1. copilot secret init --name HATCHET_CLIENT_TOKEN --overwrite
1. disable/enable TLS
1. copilot svc init --name hatchet-worker
1. make docker-login
1. copilot svc deploy --name hatchet-worker --env prod
