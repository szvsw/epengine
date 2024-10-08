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

        theme default
    }

}
