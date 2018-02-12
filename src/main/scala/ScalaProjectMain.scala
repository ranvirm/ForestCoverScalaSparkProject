import MachineLearning.{RandomForestModel, RandomForestCVModel}
import FeatureEngineering.KmeansBestBucketsCalculator
import DataWrangler.WrangledData
import org.apache.spark.sql.functions._

// Use this to run the entire project by calling the most downstream method/s

object ScalaProjectMain {

  def main(args: Array[String]): Unit = {

    val inputData = RandomForestModel()
      .select("label", "predictedLabel")

    val outputData = inputData
      .withColumn("Outcome", when(inputData("label")===inputData("predictedLabel"), "Correct").otherwise("Incorrect"))

    outputData
      .groupBy("predictedLabel", "Outcome")
      .agg(count("Outcome"))
      .show(20)



    /**
    // run to manually determine number of optimal buckets to split continuous columns into
    val testColumns = Array[String](
      "Elevation",
      "Aspect",
      "Slope",
      "Horizontal_Distance_To_Hydrology",
      "Vertical_Distance_To_Hydrology",
      "Horizontal_Distance_To_Roadways",
      "Hillshade_9am",
      "Hillshade_Noon",
      "Hillshade_3pm",
      "Horizontal_Distance_To_Fire_Points"
    )

    for(column <- testColumns) {

      KmeansBestBucketsCalculator(WrangledData(), column)

    }

      */

  }

}
