# Introduction

[DBSCAN](https://en.wikipedia.org/wiki/DBSCAN) on Spark v1.6.3.

# Usage

1. Clone this repo;

1. Package with sbt: `sbt package`.
   The artifact is *target/scala-2.10/dbscan_2.10-1.0.jar*;

1. Upload the artifact to Spark server (test on Spark 1.6.3);

1. Run on Spark server:
   `bin/spark-submit --class com.dhcc.avatar.aang.trans.steps.dbscan.lib.DBScan ~/dbscan_2.10-1.0.jar`.
