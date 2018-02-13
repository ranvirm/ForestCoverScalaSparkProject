import DataWrangler.WrangledData
import org.apache.spark.ml.feature.{Bucketizer, QuantileDiscretizer, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.sql.functions.{split, when}

// Do your feature engineering here

object FeatureEngineering {

  // define kmeans model to use to find best number of buckets to split continuous features into
  def KmeansBestBucketsCalculator(dataFrame: DataFrame, column: String): Unit = {

    val inputData = dataFrame.select(column)

    // define feature vector assembler
    val featureAssembler = new VectorAssembler()
      .setInputCols(inputData.columns)
      .setOutputCol("features")

    val clusterData = featureAssembler.transform(inputData)

    // set max number of centers to test for equal to square root of number of distinct values
    val maxCenters = math.sqrt(inputData.select(column).distinct().count()).toInt

    // create array of centers to test
    val testCenters = (2 to maxCenters).toArray

    // create mutable array to store wssse values
    var wssseArray = Array[Double]()

    // run kmeans with each k in test centers, compute wssse and print
    for(centers <- testCenters){

      // define kmeans estimator
      val kmeans = new KMeans().setK(centers).setSeed(1L)

      // fit kmeans model to data
      val model = kmeans.fit(clusterData)

      // calculate within set sum of squared errors
      val wssse = model.computeCost(clusterData)

      // add current wssse to wssse array
      wssseArray :+=  wssse

    }

    println("+++++++++++++++++++++++++++++++++")
    println(s"Calculating for column : $column")
    println("+++++++++++++++++++++++++++++++++")
    println("K, WSSSE")

    // print k and corresponding change in wssse score
    for(i <- testCenters.indices){

      // do not compute for centers equal to 2
      if(i > 0) {

        val wssseChange = math.log10(math.abs(wssseArray(i) - wssseArray(i-1)))

        val k = testCenters(i)

        println(s"$k,$wssseChange")

      }

    }

  }

  // function that returns a data frame with added  features
  def FeatureData(): DataFrame = {

    // function to create bins for input columns
    def ColumnDiscretizer(inputColumn: String, nBins: Int, dataFrame: DataFrame): DataFrame = {

      val discretizer = new QuantileDiscretizer()
        .setInputCol(inputColumn)
        .setOutputCol(inputColumn+"Bucket")
        .setNumBuckets(nBins)

      val outputDataFrame = discretizer.fit(dataFrame).transform(dataFrame)

      outputDataFrame.withColumn(inputColumn+"Bucket", outputDataFrame(inputColumn+"Bucket").cast("String"))//.drop(inputColumn)

    }

    // define function to create binary columns from input columns
    def Binarizer(inputColumns: Array[String], dataFrame: DataFrame): DataFrame = {

      // create mutable data frame to operate on
      var inputDataFrame = dataFrame

      for(column <- inputColumns) {

        // create array of unique values in column
        val uniqueValues = dataFrame.select(column).distinct().collect()

        for(i <- uniqueValues.indices) {

          // define name for new column as the unique value
          var colName = uniqueValues(i).get(0)

          // remove special characters from name for new column
          colName = colName.toString.replaceAll("[.]", "_")

          inputDataFrame = inputDataFrame
            .withColumn(column+"_"+colName, when(inputDataFrame(column)===uniqueValues(i)(0), 1).otherwise(0))

        }

        inputDataFrame = inputDataFrame.drop(column)

      }

      val outputDataFrame = inputDataFrame

      outputDataFrame

    }

    // bucket columns
    //val bucketedDataFrame = ColumnDiscretizer("Hillshade_9am", 10, ColumnDiscretizer("Hillshade_Noon", 10, ColumnDiscretizer("Hillshade_3pm", 10, WrangledData())))

    val bucketedDataFrame = ColumnDiscretizer("Aspect", 9,
      ColumnDiscretizer("Elevation", 11,
        ColumnDiscretizer("Slope", 6,
          ColumnDiscretizer("Hillshade_3pm", 9,
            ColumnDiscretizer("Hillshade_9am", 10,
              ColumnDiscretizer("Hillshade_Noon", 9,
                ColumnDiscretizer("Horizontal_Distance_To_Hydrology", 10,
                  ColumnDiscretizer("Horizontal_Distance_To_Roadways", 14,
                    ColumnDiscretizer("Horizontal_Distance_To_Fire_Points", 12,
                      ColumnDiscretizer("Vertical_Distance_To_Hydrology", 8, WrangledData()))))))))))

    // convert bucketed columns to binary
    //Binarizer(Array[String]("Hillshade_9amBucket", "Hillshade_NoonBucket", "Hillshade_3pmBucket"), bucketedDataFrame)

    Binarizer(Array[String](
      "AspectBucket",
      "Hillshade_9amBucket",
      "Hillshade_NoonBucket",
      "Hillshade_3pmBucket",
      "ElevationBucket",
      "SlopeBucket",
      "Horizontal_Distance_To_HydrologyBucket",
      "Horizontal_Distance_To_RoadwaysBucket",
      "Horizontal_Distance_To_Fire_PointsBucket",
      "Vertical_Distance_To_HydrologyBucket"
    ), bucketedDataFrame)

  }

}
