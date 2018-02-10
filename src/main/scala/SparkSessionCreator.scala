import org.apache.spark.sql.SparkSession

// Create or retrieve a spark session here
// Change the master to 'yarn' when running on a cluster

object SparkSessionCreator {

  def Spark(): SparkSession = {

    SparkSession.builder().master("local[4]").appName("CoverTypeScalaSparkProject").getOrCreate()

  }

}
