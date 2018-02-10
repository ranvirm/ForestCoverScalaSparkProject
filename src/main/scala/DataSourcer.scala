import SparkSessionCreator.Spark
import org.apache.spark.sql.DataFrame

// load all data here - do not do any data cleaning or transformations here - keep it raw
// add additional methods for individual datasets

object DataSourcer {

  def RawData(): DataFrame = {

    // get spark session
    val spark = Spark()

    // read data from parquet
    spark
      .read
      .parquet("./src/main/resources/covertypedata/")

  }

}
