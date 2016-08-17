package site.eircdoug.myspark.classifier

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by ericdoug on 16-8-17.
  */
object DecisionTree_classifier {

  def main(args: Array[String]) {

    val conf = new SparkConf()
      .setAppName("decisiontree_classifier")
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
      LabeledPoint(label, Vectors.dense(features))
    }

    data.cache()
    val numData = data.count()
    println("trainning data number: " + numData)

    /******************************
      *    构建DecisionTree模型    *
      *****************************/
    val numIterations = 10
    val maxTreeDepth = 10
    val decisiontreeModel = DecisionTree.train(data, Algo.Classification, Entropy, maxTreeDepth)



    /***********************
      *     模型验证        *
      **********************/
    val dataPoint = data.first()
    val prediction = decisiontreeModel.predict(dataPoint.features)

    println("The prediction:" + prediction)
    println("The truth:" + dataPoint.label)

    // 正确率
    val decisiontreeTotalCorrect = data.map { point =>
      if(decisiontreeModel.predict(point.features) == point.label) 1 else 0
    }.sum()

    val decisiontreeAccuracy = decisiontreeTotalCorrect / data.count

    println("Accuracy:" + decisiontreeAccuracy)

    if(sc != null) {
      sc.stop()
    }

  }

}
