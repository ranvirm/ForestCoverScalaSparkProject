import DataWrangler.WrangledData
import org.apache.spark.ml.feature.{Bucketizer, QuantileDiscretizer}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{split, when}

// Do your feature engineering here

object FeatureEngineering {

  // function that returns a data frame with added  features
  def FeatureData(): DataFrame = {

    // function to create bins for input columns
    def ColumnDiscretizer(dataFrame: DataFrame, inputColumn: String, nBins: Int): DataFrame = {

      val discretizer = new QuantileDiscretizer()
        .setInputCol(inputColumn)
        .setOutputCol(inputColumn+"Bucket")
        .setNumBuckets(nBins)

      discretizer.fit(dataFrame).transform(dataFrame).drop(inputColumn)

    }

    ColumnDiscretizer(ColumnDiscretizer(ColumnDiscretizer(WrangledData(), "Hillshade_3pm", 10), "Hillshade_Noon", 10), "Hillshade_9am", 10)

  }

}
