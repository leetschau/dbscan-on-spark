package com.dhcc.dbscan

/**
  * Created by leo on 17-6-5.
  */
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.functions.{lit, udf}
import scala.math.{pow, sqrt}

case class Node(ID: Int, X: Double, Y: Double, classified: Int)

class DBScan(coll: DataFrame, eps: Double, minPoints: Int) extends Serializable {
  def run(sqlContext: SQLContext): DataFrame = {
    import sqlContext.implicits._
    val noise = -2
    val unclassified = -1
    val marked = coll.withColumn("classified", lit(unclassified))
    val curNode = marked.where("classified < 0").limit(1).as[Node].first()
    def markNeighbours(points: DataFrame, node: Node): DataFrame = {
      val dist = udf((x: Double, y: Double, x0: Double, y0: Double) => (sqrt(pow(x - x0, 2) + pow(y - y0, 2))))
      val withDist = marked.withColumn("dist", dist($"X", $"Y", lit(curNode.X), lit(curNode.Y)))
      val setCluster = udf((old_class: Int, dist: Double, clusterID: Int ) => if (dist < eps) clusterID else old_class )
      val setNoise = udf((old_class: Int, dist: Double) => if (dist < eps) noise else old_class )
      val isNoise = withDist.where($"dist" < eps).count < minPoints
      withDist.withColumnRenamed("classified", "old_class").
        withColumn("classified", if (isNoise) setNoise($"old_class", $"dist")
        else setCluster($"old_class", $"dist", lit(curNode.ID))).
        select("ID", "X", "Y", "classified")
    }
    val newColl = markNeighbours(marked, curNode)
    val seeds = newColl.filter($"ID" !== curNode.ID).filter($"classified" === curNode.ID)
    seeds
  }
}

object DBScan {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("DBSCAN").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    val testData = Seq((0, 1.0, 1.1), (1, 1.2, 0.8), (2, 0.8, 1.0), (3, 3.7, 4.0), (4, 3.9, 3.9),
      (5, 3.6, 4.1), (6, 10.0, 10.0), (7, 1.1, 0.9), (8, 10.1, 9.9))
    val inp = sqlContext.createDataFrame(testData).toDF("ID", "X", "Y")
    val dbscan = new DBScan(inp, 0.5, 2)
    val res = dbscan.run(sqlContext)
    res.show()

  }
}
