package site.eircdoug.myspark.classifier

import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by ericdoug on 16-8-2.
  */
object Logistic_classifier {

  def accuracy(data:RDD[LabeledPoint], model:LogisticRegressionModel) {


  }

  def main(args: Array[String]) {
    // 数据处理
    val conf = new SparkConf()
      .setAppName("logistic_classifier")
      .setMaster("local")
    val sc = new SparkContext(conf)

    val train_file = "/home/ericdoug/datas/kaggle/stumbleupon/train_noheader.csv"

    val rawData = sc.textFile(train_file)
    val records = rawData.map(line => line.split("\t"))
    val first_record = records.first()
    first_record.foreach(println)

    /***********************************
      *         数据清洗和处理           *
      **********************************/

    val data = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map(d => if(d == "?") 0.0 else d.toDouble)
        .map(d => if(d < 0) 0.0 else d)
      LabeledPoint(label, Vectors.dense(features))
    }

    data.cache()
    val numData = data.count()
    println("trainning data number: " + numData)

    /***************************
      *    构建Logistic模型     *
      **************************/
    val numIterations = 10
    val lrModel = LogisticRegressionWithSGD.train(data, numIterations)


    /***********************
      *     模型验证        *
      **********************/
    val dataPoint = data.first()
    val prediction = lrModel.predict(dataPoint.features)

    println("The prediction:" + prediction)
    println("The truth:" + dataPoint.label)

    // 正确率
    val lrTotalCorrect = data.map { point =>
      if(lrModel.predict(point.features) == point.label) 1 else 0
    }.sum()

    val lrAccuracy = lrTotalCorrect / data.count

    println("Accuracy:" + lrAccuracy)

    if(sc != null) {
      sc.stop()
    }
  }
}