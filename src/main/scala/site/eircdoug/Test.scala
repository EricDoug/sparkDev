package site.eircdoug

import org.apache.spark.SparkContext

/**
  * Created by ericdoug on 16-8-1.
  */

class Test {

}


object Test {
  def main(args: Array[String]) {
    val sc = new SparkContext("local","Test")

    //  val data = sc.parallelize(List(1,2,3,4))
    //
    //  val result = data.map(_ * 2)

    //  result.foreach(println)
    val text = sc.textFile("hdfs://localhost:9000/user/ericdoug/README.md")
    val result = text.count()
    println("--------------------------------")
    println(result)
    println("---------------------------------")
    sc.stop()
  }
}