import DataSourcer.RawData
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.QuantileDiscretizer
import org.apache.spark.sql.functions.{mean, round}

// Clean raw data frames here - format columns, replace missing values etc.

object DataCleaner {

  // function to produce a clean data frame from a raw data frame
  def CleanData(): DataFrame = {

    // function to convert column to type double
    def DoubleType(dataFrame: DataFrame, column: String): DataFrame = {

      dataFrame.withColumn(column, dataFrame(column).cast("Double"))

    }

    // function to convert column to type string
    def StringType(dataFrame: DataFrame, column: String): DataFrame = {

      dataFrame.withColumn(column, dataFrame(column).cast("String"))

    }

    StringType(DoubleType(DoubleType(DoubleType(RawData(), "Hillshade_3pm"), "Hillshade_Noon"), "Hillshade_9am"), "Cover_Type")

  }


}
