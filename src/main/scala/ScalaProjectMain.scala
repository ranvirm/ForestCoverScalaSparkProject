import OutputSaver.RandomForestPipeLineModelSave
import FeatureEngineering.KmeansBestBucketsCalculator

// Use this to run the entire project by calling the most downstream method/s

object ScalaProjectMain {

  def main(args: Array[String]): Unit = {

    RandomForestPipeLineModelSave()

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
