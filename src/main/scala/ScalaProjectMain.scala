import MachineLearning.MachineLearningOutput
import org.apache.spark.sql.functions._

// Use this to run the entire project by calling the most downstream method/s

object ScalaProjectMain {

  def main(args: Array[String]): Unit = {

    val inputData = MachineLearningOutput()
      .select("label", "predictedLabel")

    val outputData = inputData
      .withColumn("Outcome", when(inputData("label")===inputData("predictedLabel"), "Correct").otherwise("Incorrect"))

    outputData
      .groupBy("predictedLabel", "Outcome")
      .agg(count("Outcome"))
      .show(20)

  }

}
