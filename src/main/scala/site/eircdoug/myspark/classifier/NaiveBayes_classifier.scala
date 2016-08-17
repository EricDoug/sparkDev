package site.eircdoug.myspark.classifier

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.NaiveBayes

/**
  * Created by ericdoug on 16-8-17.
  */
object NaiveBayes_classifier {

  def main(args: Array[String]) {

    val conf = new SparkConf()
      .setAppName("naivebayes_classifier")
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

    val nbdata = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map(d => if(d == "?") 0.0 else d.toDouble)
        .map(d => if(d < 0) 0.0 else d)
      LabeledPoint(label, Vectors.dense(features))
    }

    nbdata.cache()
    val numData = nbdata.count()
    println("trainning data number: " + numData)

    /*****************************
      *    构建NaiveBayes模型     *
      ***************************/
    val numIterations = 10
    val naivebayesModel = NaiveBayes.train(nbdata, numIterations)


    /***********************
      *     模型验证        *
      **********************/
    val dataPoint = nbdata.first()
    val prediction = naivebayesModel.predict(dataPoint.features)

    println("The prediction:" + prediction)
    println("The truth:" + dataPoint.label)

    // 正确率
    val naivebayesTotalCorrect = nbdata.map { point =>
      if(naivebayesModel.predict(point.features) == point.label) 1 else 0
    }.sum()

    val naivebayesAccuracy = naivebayesTotalCorrect / nbdata.count

    println("Accuracy:" + naivebayesAccuracy)

    if(sc != null) {
      sc.stop()
    }
  }
}
