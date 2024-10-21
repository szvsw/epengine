workspace "Name" "Description" {
    !identifiers hierarchical
    model {
        properties {
            "structurizr.groupSeparator" "/"
        }

        extSystems = group "External Systems" {
            externalQueue = softwareSystem "Hatchet" {
                tags "Database", "External"
                db = container "Database" {
                    technology "Postgres"
                    tags "Database"
                }
                q = container "Queue" {
                    technology "RabbitMQ"
                    tags "Queue"
                }
                engine = container "Engine" {
                    technology "Go"
                    tags "API"
                }
                api = container "API" {
                    technology "Go"
                    tags "API"
                }
                fe = container "Frontend" {
                    technology "React/Next"
                    tags "Browser"
                }
                caddy = container "Caddy" {
                    technology "Caddy"
                    tags = "API"
                }

                caddy -> fe "serves" {
                    tags "Browser"
                }
                caddy -> api "serves" {
                    tags "API"
                }
                fe -> api "requests data from" {
                    tags "API"
                }
                engine -> q "populates/consumes" {
                    tags "DataPull"
                }
                engine -> db "populates/consumes" {
                    tags "DataPull"
                }
                api -> db "populates/consumes" {
                    tags "DataPull"
                }
            }

            overture = softwareSystem "Overture Maps" {
                tags "Database", "External"
            }
        }

        app = group "AWS" {
            workers = softwareSystem "Worker Nodes" {
                tags "Model"
                simWorker = container "Simulation Worker" {
                    technology "Python"
                    tags "Engine"
                }
                gisWorker = container "GIS Worker" {
                    technology "Python"
                    tags "Engine"
                }
            }


            data = softwareSystem "Storage" {
                tags "Database"
                bucket = container "Bucket" {
                    technology "S3"
                    tags "Database"
                }
                db = container "Database" {
                    technology "Supabase"
                    tags "Database"
                }
            }

            workers -> data "r/w from" {
                tags "DataPull"
            }

            workers.simWorker -> data.bucket "r/w from" {
                tags "DataPull"
            }

            workers.gisWorker -> data.bucket "r/w from" {
                tags "DataPull"
            }
        }

        workers.simWorker -> externalQueue "consumes" {
            tags "DataPull"
        }
        workers.gisWorker -> externalQueue "consumes" {
            tags "DataPull"
        }
        workers.simWorker -> externalQueue.engine "pulls jobs from" {
            tags "DataPull"
        }
        workers.gisWorker -> externalQueue.engine "pulls jobs from" {
            tags "DataPull"
        }
        workers.gisWorker -> overture "requests GIS data from" {
            tags "DataPull"
        }

        user = Person "User" {
            tags "Researcher"
        }

        user -> externalQueue "schedules jobs" {
            tags "DataPull"
        }
        user -> externalQueue.caddy "schedules jobs" {
            tags "Browser"
        }

        prod = deploymentEnvironment "Production" {
            deploymentNode "AWS" {
                s3 = infrastructureNode "Bucket" {
                    technology "S3"
                    tags "Amazon Web Services - Simple Storage Service"
                }
                vpc = deploymentNode "VPC" {
                    tags "Amazon Web Services - Virtual Private Cloud"
                    pub = deploymentNode "Public Subnets" {
                        ecs = deploymentNode "Cluster" {
                            technology "ECS"
                            tags "Amazon Web Services - Elastic Container Service"
                            hatchetEngine = deploymentNode "Hatchet Engine" "The distributed queueing engine." "Fargate" {
                                tags "Amazon Web Services - Fargate"
                                engine = containerInstance externalQueue.engine
                            }
                            hatchetApp = deploymentNode "Hatchet Admin" {
                                technology "Fargate"
                                tags "Amazon Web Services - Fargate"
                                apiInstance = containerInstance externalQueue.api
                                feInstance = containerInstance externalQueue.fe
                            }
                            workerNode = deploymentNode "Worker Nodes"{
                                technology "Fargate"
                                tags "Amazon Web Services - Fargate"
                                worker = containerInstance workers.simWorker
                                instances 8
                            }
                            efs = infrastructureNode "EFS" {
                                technology "EFS"
                                tags "Amazon Web Services - EFS"
                            }
                            hatchetApp.apiInstance -> efs "reads config from" {}
                            hatchetEngine.engine -> efs "reads config from" {}

                        }

                        lb = infrastructureNode "Load Balancer" {
                            tags "Amazon Web Services - Elastic Load Balancing"
                            technology "ALB"
                        }
                        lb -> ecs.hatchetApp.apiInstance "routes to" {
                            tags "API"
                        }
                        lb -> ecs.hatchetApp.feInstance "routes to" {
                            tags "Browser"
                        }
                    }
                    deploymentNode "Private Subnets" {
                        deploymentNode "Aurora Serverless" "The Queue database" "RDS" {
                            tags "Amazon Web Services - RDS"
                            containerInstance externalQueue.db
                        }
                        deploymentNode "RabbitMQ" "The Queue" "AmazonMQ" {
                            tags "Amazon Web Services - MQ"
                            containerInstance externalQueue.q
                        }
                    }
                }
                vpc.pub.ecs.workerNode.worker -> s3 "reads/writes from" {
                    tags "DataPull"
                }
            }
        }
    }

    views {

        systemLandscape sys {
            include *
        }
        systemContext workers {
            include *
        }

        container workers {
            include *
        }

        systemContext externalQueue {
            include *
        }

        container externalQueue {
            include *
        }

        deployment * prod {
            include *
            // autolayout
        }

        styles {
            element Element {
                fontsize 32
            }


            // Object styles
            element "Group" {
                strokeWidth 4
                fontSize 32
                color #909090
            }
            element "Software System" {
                color #505050
            }
            element "Container" {
                color #505050
            }

            // People Styles
            element "User" {
                background #DEDAF4
                color #000000
            }
            element "Scientist" {
                background #D4E1DE
                color #000000
            }
            element "Local Expert" {
                background #FFD6A5
                color #000000
            }


            // Tag Styles
            element "Database" {
                shape cylinder
                background #7ec4Cf
            }
            element "Queue" {
                shape cylinder
                background #c47eCf
            }
            element "API" {
                shape pipe
                background #9ab7d3
            }

            element "Model" {
                shape pipe
                background #ffadad
            }

            element "Browser" {
                background #9ab7d3
                shape WebBrowser
            }
            element "External" {
                background #AAAAAA
            }

            // Relationship Styles
            relationship Relationship {
                dashed False
                routing Curved
                style dotted
                fontSize 28
            }
            relationship "contribution" {
                style dashed

            }
            relationship "DataPull" {
                style dotted
            }
            relationship "EngineCall" {
                style solid
            }

        }

        theme https://static.structurizr.com/themes/amazon-web-services-2023.01.31/theme.json
        //icons here: https://github.com/structurizr/themes/blob/master/amazon-web-services-2023.01.31/icons.json
    }

}
