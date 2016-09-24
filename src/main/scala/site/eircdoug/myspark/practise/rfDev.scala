package site.eircdoug.myspark.practise

import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, RandomForestModel}
import org.apache.spark.rdd.RDD

/**
  * Created by ericdoug on 16-9-25.
  */
object rfDev {

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

    val sc = new SparkContext("local", "RandomForest")

    val data_file = "/home/ericdoug/datas/spark/covtype.data"
    val data = data_handler(sc, data_file)

    val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache()
    cvData.cache()
    testData.cache()


    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int,Int]()
    val numTrees = 3
    val featureSubsetStrategy = "auto"
    val impurity = "variance"
    val maxDepth = 4
    val maxBins = 32
    val model = RandomForest.trainRegressor(trainData, categoricalFeaturesInfo, numTrees, featureSubsetStrategy,
      impurity, maxDepth, maxBins)

    val labelesAndPredictions = testData.map{ point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    val testMSE = labelesAndPredictions.map{ case(v,p) =>
      math.pow((v-p),2)
    }.mean()

    println("Test Mean Squared Error = " + testMSE)
    println("Learned regression forest model:\n" + model.toDebugString)
    model.save(sc, "rfModel")
    // val sameModel = RandomForestModel.load(sc, "rfModel")

  }
}
