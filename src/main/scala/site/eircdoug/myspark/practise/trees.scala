package site.eircdoug.myspark.practise

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD

/**
  * Created by ericdoug on 16-9-24.
  */
object Trees {

  def data_handler(sc: SparkContext, data_file: String): RDD[LabeledPoint] = {
    val datas = sc.textFile(data_file)
      .map { line =>
        val items = line.split(',').map(_.toDouble)
        val features = Vectors.dense(items.init)
        val label = items.last - 1
        (label, features)
      }

    datas.map(line =>
      LabeledPoint(line._1, line._2)
    )
  }


  def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
    val predictionAndLabels = data.map(example =>
      (model.predict(example.features), example.label)
    )
    new MulticlassMetrics(predictionAndLabels)
  }

  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.apache.jetty.server").setLevel(Level.OFF)

//    val sparkHome = "/home/ericdoug/spark/spark-1.6.0-bin-hadoop1"
//    val master = "local"
//    val conf = new SparkConf()
//      .setMaster(master)
//      .setSparkHome(sparkHome)
//      .setAppName("Tree")
//      .set("spark.executor.memory", "2g")
//
//    val sc = new SparkContext(conf)

    val sc = new SparkContext("local","Trees")

    val data_file = "/home/ericdoug/datas/spark/covtype.data"
    val data = data_handler(sc, data_file)

    val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache()
    cvData.cache()
    testData.cache()

    val model = DecisionTree.trainClassifier(
      trainData, 7, Map[Int, Int](), "gini", 4, 100
    )

    val metrics = getMetrics(model, cvData)

    if (sc != null) {
      sc.stop()
    }
  }
}
