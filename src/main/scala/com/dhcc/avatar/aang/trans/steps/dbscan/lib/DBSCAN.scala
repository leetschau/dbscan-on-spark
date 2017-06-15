package com.dhcc.avatar.aang.trans.steps.dbscan.lib

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.functions.{lit, udf}

import scala.annotation.tailrec
import scala.math.{pow, sqrt}

/**
  * Created by leo on 17-6-5.
  * DBSCAN: Density based Spatial Clustering of Applications with Noise
  * Input: DataFrame
  * Output: DataFrame
  * Parameters:
  *   eps:
  *   minPoints:
  */
case class Node(ID: Int, X: Double, Y: Double, classified: Int)

class DBScan(spc: SparkContext, coll: DataFrame, eps: Double, minPoints: Int) extends Serializable {
  @transient val sc = spc
  val Noise = -1
  val Unclassified = -10
  val sqlContext = new SQLContext(sc)
  import sqlContext.implicits._
  def run(): DataFrame = {
    val marked = coll.withColumn("classified", lit(Unclassified)).withColumn("seed", lit(0))
    markCluster(marked)
  }

  @tailrec
  private def markCluster(points: DataFrame): DataFrame = {
    val untested = points.filter($"classified" < Noise).limit(1)
    if (untested.count() < 1) {
      points
    } else {
      val curNode = untested.as[Node].first()
      val markCurrent = udf((id: Int, seed: Int) => if (id == curNode.ID) 1 else seed)
      val markCurNode = points.withColumn("seed", markCurrent($"ID", $"seed"))
      val newCluster = extendNeighbours(markCurNode)
      markCluster(newCluster)
    }
  }

  def markNeighbours(points: DataFrame, node: Node): DataFrame = {
    val dist = udf((x: Double, y: Double, x0: Double, y0: Double) => (sqrt(pow(x - x0, 2) + pow(y - y0, 2))))
    val withDist = points.withColumn("dist", dist($"X", $"Y", lit(node.X), lit(node.Y)))

    val setCluster = udf((classified: Int, dist: Double) =>
    {
      if (dist < eps && node.classified >= 0) node.classified
      else if (dist < eps && node.classified < 0) node.ID
      else classified
    }
    )
    val setNoise = udf((classified: Int, dist: Double) => if (dist < eps) Noise else classified )
    val isNoise = withDist.filter($"dist" < eps).filter($"dist" >= 0).count < minPoints
    val setSeed = udf((id: Int, dist: Double, seed: Int, classified: Int) =>
      if (classified < 0 && dist < eps && id != node.ID) 1
      else if (id == node.ID) 0
      else seed
    )
    val markSeed = if (isNoise) withDist else withDist.withColumn("seed", setSeed($"ID", $"dist", $"seed", $"classified"))
    val new_classified = markSeed.withColumn("classified",
      if (isNoise) setNoise($"classified", $"dist") else setCluster($"classified", $"dist"))
    new_classified
  }

  @tailrec
  private def extendNeighbours(points: DataFrame): DataFrame = {
    val dataLen = points.filter($"seed" > 0).count()
    if (dataLen == 0) {
      points
    } else {
      val firstSeed = points.filter($"seed" > 0).limit(1).as[Node].first()
      val firstMarked = markNeighbours(points, firstSeed)
      firstMarked.cache()
      extendNeighbours(firstMarked)
    }
  }
}

object DBScan {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("DBSCAN").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    val testData = Seq((0, 1.0, 1.1), (1, 2.0, 1.0), (2, 0.9, 1.0), (3, 3.7, 4.0), (4, 3.9, 3.9),
      (5, 3.6, 4.1), (6, 10.0, 10.0), (7, 2.9, 1.0), (8, 10.1, 9.9), (9, 3.9, 1.0))
    val inp = sqlContext.createDataFrame(testData).toDF("ID", "X", "Y")
    val dbscan = new DBScan(sc, inp, 1.5, 2)
    val res = dbscan.run()
    println("Final result:")
    res.show()
  }
}
